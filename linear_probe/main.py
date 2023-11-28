# %%
import sys

sys.path.append("../")

import os

import torch
import torch.nn as nn

import clip
from linear_probe.args import parse_args
from linear_probe.dataset import get_combined_loader
from linear_probe.helpers import *
from linear_probe.network import LinearProbe
from linear_probe.trainer import train

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
def main(args):
    """Load model"""

    backbone = "RN50"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(device)

    model, preprocess = clip.load(backbone, device=device)
    model = model.visual

    model.to(device)

    print(f"CLIP-{backbone} lodad")

    # Load data

    data_loader = get_combined_loader(
        root_dir=args.root_data,
        transform=preprocess,
        batch_size=args.batch_size,
        subset_samples=args.subset_samples,
        pin_memory=True,
    )

    print("Dataset loaded")

    # import pdb

    # pdb.set_trace()
    # %%
    # Linear probing model

    if args.obj == "layer2_3_relu3":
        in_dim = 512 * 28 * 28
    elif args.obj == "layer1_3_relu3":
        in_dim = 256 * 56 * 56
    elif args.obj == "layer4_2_relu2":
        in_dim = 512 * 7 * 7
    elif args.obj == "layer3_5_relu3":
        in_dim = 1024 * 14 * 14
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
