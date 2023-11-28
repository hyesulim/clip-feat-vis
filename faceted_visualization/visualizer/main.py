import ast
import os
import pprint
import sys
from typing import Callable, Dict

sys.path.append("../")

import torch
from faceted_visualization.visualizer import args, hook, image, constants
from faceted_visualization.visualizer import logger as logger_
import clip
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger()
run_id = logger_.run_id


def get_probe_weights(checkpoint_directory: str, checkpoint_file_name: str) -> torch.Tensor:
    file_name = os.path.join(checkpoint_directory, checkpoint_file_name)
    logger.info("Retrieving weights of linear probe [ path = %s ]", file_name)
    linear_probe = torch.load(file_name, map_location=device)
    if "linear.weight" in linear_probe.keys():
        return linear_probe["linear.weight"][0]
    elif "weight" in linear_probe.keys():
        return linear_probe["weight"][0]


def get_model(model_name) -> torch.nn.Module:
    logger.info("Loading CLIP model [ %s ].", model_name)
    if model_name in clip.available_models():
        model, transforms = clip.load(model_name, device=device)
        model = model.visual
        logger.info("Finished loading model [ %s ]", model_name)
        return model
    else:
        raise RuntimeError("Unable to locate CLIP model. Possible values are {!s}".format(clip.available_models()))


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


def optimize(num_iterations: int, image_function: Callable, model: torch.nn.Module, channel: int, objective: str,
             layer: str, linear_probe_layer: str, probe_weights: torch.Tensor, optimizer: torch.optim.Optimizer,
             model_hook: Callable):
    logger.info("Starting optimization process [ run_id = %s ]...", run_id)

    for j in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            model.forward(image_function())

            f_x = model_hook(layer, True)
            g_x = model_hook(linear_probe_layer, True)

            gradients = torch.autograd.grad(f_x.output, g_x.output, grad_outputs=torch.ones_like(f_x.output),
                                            retain_graph=True)[0]
            if objective == 'neuron':
                neuron_x = f_x.output.shape[2] // 2
                neuron_y = f_x.output.shape[3] // 2
                loss = -(f_x.output[0, channel, neuron_x, neuron_y])
            else:
                loss = -(f_x.output[0, channel, :, :].mean())
            loss -= torch.dot(torch.mul(g_x.output, gradients).flatten().type(torch.FloatTensor).to(device),
                              probe_weights.to(device))

            loss.backward()
            if j % 100 == 0:
                logger.info("Epoch: %d/%d: Loss = %.7f :: Learning Rate = %.7f", j + 1, num_iterations, loss,
                            optimizer.param_groups[0]['lr'])
            return loss

        optimizer.step(closure)


def save_results(image_array: torch.Tensor, config: Dict):
    output_directory = os.path.join(config[constants.OUTPUT_DIRECTORY], run_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, "output.jpg")
    logger.info("Saving image [ path = %s ]", output_path)
    pil_image = image.convert_to_PIL(image_array)
    pil_image.save(output_path)


def orchestrate(config, save_to_file: bool = False):
    if save_to_file:
        logger_.add_file_handler()
    model = get_model(config[constants.MODEL])

    model_hook = hook.register_hooks(model)

    params, image_f = image.generate_img(w=config[constants.IMAGE_WIDTH],
                                         h=config[constants.IMAGE_HEIGHT],
                                         decorrelate=config[constants.IMG_DECORRELATE],
                                         fft=config[constants.IMG_FFT])

    optimizer = get_optimizer(parameters=params, optimizer_name=config[constants.OPTIMIZER],
                              learning_rate=config[constants.LEARNING_RATE])

    probe_weights = get_probe_weights(config[constants.PROBES_DIRECTORY], config[constants.CHECKPOINT_FILENAME])

    optimize(num_iterations=config[constants.NUMBER_OF_ITERATIONS],
             image_function=image_f,
             model=model,
             channel=config[constants.CHANNEL],
             objective=config[constants.OBJECTIVE],
             layer=config[constants.VISUALIZATION_LAYER],
             linear_probe_layer=config[constants.LINEAR_PROBE_LAYER],
             probe_weights=probe_weights,
             optimizer=optimizer,
             model_hook=model_hook)

    save_results(image_f(), config)


def combine_properties(shell_arguments: Dict, properties: dict) -> Dict:
    merged = dict()
    # Shell arguments get higher priority
    for k, v in properties.items():
        merged[k] = v
    for k, v in shell_arguments.__dict__.items():
        if v is not None:
            merged[k] = v
    return merged


if __name__ == "__main__":
    arguments = args.parse_args()

    cwd = os.path.dirname(__file__)

    with open(os.path.join(cwd, "config", "run_configs.json")) as f:
        properties = ast.literal_eval(f.read())
    properties = combine_properties(arguments, properties)
    logger.info("Load properties = \n%s", pprint.pformat(properties))
    try:
        orchestrate(config=properties, save_to_file=False)
    except Exception as e:
        logger.exception("Something went wrong.")
