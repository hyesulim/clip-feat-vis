import argparse
import ast
import os
import os.path as osp
import pprint
import sys
from typing import Callable, Dict, Tuple

import torch
import torchvision.transforms.transforms
import torchvision.transforms.v2

import cli, hook, image, constants, render, wb, helpers
import logger as logger_
import clip
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger()
run_id = logger_.run_id


def get_probe_weights(model_location: str, device: str) -> torch.Tensor:
    logger.info("Retrieving weights of linear probe [ path = %s ]", model_location)
    linear_probe = torch.load(model_location, map_location=device)
    if "linear.weight" in linear_probe.keys():
        return linear_probe["linear.weight"][0]
    elif "weight" in linear_probe.keys():
        return linear_probe["weight"][0]



def get_model(
    model_name, ckpt_path: str, device="cpu"
) -> Tuple[torch.nn.Module, torchvision.transforms.transforms.Compose]:
    logger.info("Loading CLIP model [ %s ].", model_name)
    if model_name in clip.available_models():
        model, _, transforms = clip.load(model_name, device=device, jit=False)
        if len(ckpt_path) > 1:
            model = torch.load(ckpt_path)
            model = model.image_encoder.model.visual
            model.to(device)
        else:
            model = model.visual

        logger.info("Finished loading model [ %s ]", model_name)
        # transforms[2, 3] is to convert image to tensor. This is not required
        del transforms.transforms[2]
        del transforms.transforms[2]
        logger.info("Finished loading transforms [ %s ]", transforms)
        return model, transforms.transforms
    else:
        raise RuntimeError(
            "Unable to locate CLIP model. Possible values are {!s}".format(
                clip.available_models()
            )
        )


def get_optimizer(parameters, optimizer_name: str, learning_rate: float):
    if optimizer_name == constants.ADAM:
        logger.info("Creating Adam optimizer with params [ lr = %.5f ]", learning_rate)
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == constants.ADAM_W:
        logger.info("Creating AdamW optimizer with params [ lr = %.5f ]", learning_rate)
        return torch.optim.AdamW(parameters, lr=learning_rate)
    elif optimizer_name == constants.SGD:
        logger.info("Creating SGD optimizer with params [ lr = %.5f ]", learning_rate)
        return torch.optim.SGD(parameters, lr=learning_rate)


def orchestrate(
    config: Dict, wandb_object: wb.WandB = None, save_to_file: bool = False, device="cpu") -> Callable:
    if save_to_file:
        logger_.add_file_handler(config)
    model, clip_transforms = get_model(model_name=config[constants.MODEL],
                                       ckpt_path=config[constants.CKPT_PATH],
                                       device=device)


    model_hook = hook.register_hooks(model)
    helpers.set_seed(config.get(constants.RANDOM_SEED, None))

    params, image_f = image.generate_img(
        w=config[constants.IMAGE_WIDTH],
        h=config[constants.IMAGE_HEIGHT],
        decorrelate=bool(config[constants.IMG_DECORRELATE]),
        fft=bool(config[constants.IMG_FFT]),
        device=device,
    )

    optimizer = get_optimizer(
        parameters=params,
        optimizer_name=config[constants.OPTIMIZER],
        learning_rate=config[constants.LEARNING_RATE],
    )

    # Find the last version among LP ckpts
    lp_dir = osp.join(config[constants.PATH_LINEAR_PROBE], config[constants.LINEAR_PROBE_LAYER])
    existing_versions = [
        d
        for d in os.listdir(lp_dir)
        if os.path.isdir(os.path.join(lp_dir, d)) and d.startswith("version_")
    ]
    latest_version = max([int(v.split("_")[1]) for v in existing_versions])
    # Use 10 epoch weight by default
    lp_ckpt_path = osp.join(lp_dir, f"version_{latest_version}", 'model_checkpoint-10.pth')
    probe_weights = get_probe_weights(
        model_location=lp_ckpt_path, device=device
    )
    #probe_weights = get_probe_weights(model_location=config[constants.PATH_LINEAR_PROBE], device=device)

    transforms = image.consolidate_transforms(
        use_clip_transforms=config[constants.USE_TRANSFORMS],
        use_standard_transforms=config[constants.USE_STD_TRANSFORMS],
        clip_transforms=clip_transforms
    )
    logger.info("Final list of transforms = %s", transforms.transforms if transforms is not None else "NA")

    render.optimize(num_iterations=config[constants.NUMBER_OF_ITERATIONS],
                    transforms=transforms,
                    image_function=image_f,
                    model=model,
                    channel=config[constants.CHANNEL],
                    objective=config[constants.OBJECTIVE],
                    layer=config[constants.VISUALIZATION_LAYER],
                    linear_probe_layer=config[constants.LINEAR_PROBE_LAYER],
                    probe_weights=probe_weights,
                    optimizer=optimizer,
                    model_hook=model_hook,
                    neuron_x=config.get(constants.NEURON_X, None),
                    neuron_y=config.get(constants.NEURON_Y, None),
                    wandb_object=wandb_object,
                    run_id=run_id,
                    device=device)

    helpers.save_results(image_array=image_f(),
                         output_directory=os.path.join(config[constants.PATH_OUTPUT], run_id),
                         create_dir=True)

    if len(config[constants.CKPT_PATH]) > 1:
        ft_mod = f"_{config[constants.CKPT_PATH].split('/')[-1][:-3]}"
    else:
        ft_mod = ''


    helpers.save_results(
        image_array=image_f(),
        output_directory=os.path.join(
            config[constants.PATH_OUTPUT], config[constants.OBJECTIVE]+ft_mod, config[constants.LINEAR_PROBE_LAYER], run_id
        ),
        create_dir=True,
    )

    return image_f


if __name__ == "__main__":
    arguments = cli.parse_args()

    cwd = os.path.dirname(__file__)

    with open(arguments.__dict__[constants.PATH_CONFIG_FILE]) as f:
        properties = ast.literal_eval(f.read())
    properties = helpers.combine_properties(arguments, properties)
    logger_.add_file_handler(properties)
    logger.info("Load properties = \n%s", pprint.pformat(properties))
    wandb_object_ = None
    try:
        wandb_object_ = wb.WandB(
            enabled=properties[constants.WANDB_ENABLED],
            api_key=properties[constants.WANDB_API_KEY],
            entity=properties[constants.WANDB_ENTITY],
            project=properties[constants.WANDB_PROJECT],
            run_name=properties.get(constants.WANDB_RUN_NAME, None),
            config=properties,
        )
        orchestrate(config=properties, save_to_file=True, wandb_object=wandb_object_)
    except Exception as e:
        logger.exception("Something went wrong.")
        if wandb_object_ is not None:
            wandb_object_.close()
