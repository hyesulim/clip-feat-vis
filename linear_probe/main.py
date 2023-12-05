# %%
import sys

sys.path.append("../")

import os

import torch
import torch.nn as nn

import clip.clip as clip
from linear_probe.args import parse_args
from linear_probe.dataset import get_combined_loader
from linear_probe.helpers import *
from linear_probe.network import LinearProbe
from linear_probe.trainer import train

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# %%

IN_DIM_RN50 = {
    "layer1_2_relu": 256 * 56 * 56,
    "layer2_3_relu": 512 * 28 * 28,
    "layer3_5_relu": 1024 * 14 * 14,
    "layer4_2_relu": 512 * 7 * 7,
}
# IN_DIM_RN50 = {
#     'layer2_3_relu3': 512 * 28 * 28,
#     "layer1_3_relu3": 256 * 56 * 56,
#     'layer4_2_relu2': 512 * 7 * 7,
#     "layer3_5_relu3": 1024 * 14 * 14,
# }

IN_DIM_RN50x4 = {
    "layer1_0_conv3": 320 * 72 * 72,
    "layer1_0_relu": 320 * 72 * 72,
    "layer1_3_conv3": 320 * 72 * 72,
    "layer1_3_relu": 320 * 72 * 72,
    "layer2_5_conv3": 640 * 36 * 36,
    "layer2_5_relu": 640 * 36 * 36,
    "layer3_9_conv3": 1280 * 18 * 18,
    "layer3_9_relu": 1280 * 18 * 18,
}


def main(args):
    """Load model"""

    backbone = args.model_arch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(device)


    # model, preprocess = clip.load(backbone, device=device)
    # model = model.visual
    model, _, preprocess = clip.load(args.model_arch, device=device, jit=False)
    if len(args.ftckpt_dir) > 1:
        model = torch.load(args.ftckpt_dir)
        if args.ftckpt_dir.split('/')[-1][:-3].split('_')[1] == 'FT':
            model = model.image_encoder.model.visual
        else:
            model = model.model.visual
    else:
        model = model.visual
    print(model)
    print('turn off gradients')
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    print(f"CLIP-{backbone} lodad")

    # Load data

    data_loader = get_combined_loader(
        root_dir=args.root_data,
        transform=preprocess,
        batch_size=args.batch_size,
        subset_samples=args.subset_samples,
        pin_memory=True,
        target=args.lp_dataset,
    )

    print("Dataset loaded")

    import pdb

    #pdb.set_trace()
    # %%
    # Linear probing model

    in_dim = IN_DIM_RN50[args.obj] if args.model_arch == 'RN50' else IN_DIM_RN50x4[args.obj]
    out_dim = 1  # discriminate two datasets
    linear_probe = LinearProbe(in_dim, out_dim)
    linear_probe.to(device)

    if args.train:
        linear_probe, load_dir = train(
            model, linear_probe, data_loader, args, obj=args.obj
        )

    elif args.ckpt_dir is not None:
        load_dir = args.ckpt_dir

    loaded_model = load_model(linear_probe, load_dir)


# %%
if __name__ == "__main__":
    args = parse_args()
    main(args)
