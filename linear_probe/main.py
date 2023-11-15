# %%
import sys

sys.path.append("../")

import os

import torch
import torch.nn as nn

import clip
from linear_probe.dataset import get_combined_loader
from linear_probe.helpers import *
from linear_probe.trainer import train

# %%
""" Load model """

backbone = "RN50"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(backbone, device=device)
model = model.visual

model.to(device)

config = {
    "root_data": "/home/nas2_userH/hyesulim/Data",
    "root_code": "/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/linear_probe",
    "device": device,
    "optim": "adam",
    "lr": 1e-3,
    "num_epochs": 10,
    "obj": "layer1_2_relu3",
    "batch_size": 128,
    "subset_samples": 10000,
    "ckpt_dir": "/home/nas2_userH/hyesulim/Dev/2023/11785-f23-prj/linear_probe/logs/layer1_2_relu3/version_1"
    # "ckpt_dir": None
}

# Load data

data_loader = get_combined_loader(
    root_dir=config["root_data"],
    transform=preprocess,
    batch_size=config["batch_size"],
    subset_samples=config["subset_samples"],
)

# Linear probing model

in_dim = 256 * 56 * 56
out_dim = 2  # discriminate two datasets
linear_probe = nn.Linear(in_dim, out_dim)
linear_probe.to(device)

if config["train"]:
    linear_probe, load_dir = train(
        model, linear_probe, data_loader, config, obj=config["obj"]
    )

elif config["ckpt_dir"] is not None:
    load_dir = config["ckpt_dir"]

loaded_model = load_model(linear_probe, load_dir)
# %%
