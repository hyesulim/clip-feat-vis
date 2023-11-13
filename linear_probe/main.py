# %%
import sys
sys.path.append("../")

import torch
import torch.nn as nn

import clip
from linear_probe.dataset import get_combined_loader
from linear_probe.trainer import train

# %%
""" Load model """

backbone = "RN50"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(backbone, device=device)
model = model.visual

model.to(device)
# %%
""" Load data """

ROOT = "/home/nas2_userH/hyesulim/Data"
data_loader = get_combined_loader(root_dir=ROOT, transform=preprocess)

# %%
"""
Linear probing model
"""

in_dim = 256 * 56 * 56
out_dim = 2  # discriminate two datasets
linear_probe = nn.Linear(in_dim, out_dim)
linear_probe.to(device)

config = {
    "device": device,
    "optim": "adam",
    "lr": 1e-3,
    "num_epochs": 10,
    "obj": "layer1_2_relu3",
}

linear_probe = train(model, linear_probe, data_loader, config, obj=config["obj"])

# %%
