#%%
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
""" Load model """
import clip
backbone = "RN50"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(backbone, device=device)
model = model.visual

model.to(device)
# %%
""" Load data """
from dataset import get_combined_loader

ROOT = '/home/nas2_userH/hyesulim/Data'
data_loader = get_combined_loader(root_dir=ROOT, transform=preprocess)

#%%
""" (Optional) Visualize data """
from combined_dataset import plot_images
plot_images(data_loader)

# %%
""" (Helpers) Hooking function """
from collections import OrderedDict
from lucent.optvis.render import ModuleHook

def hook_layers(net, prefix=[]):
    if hasattr(net, "_modules"):
        for name, layer in net._modules.items():
            if layer is None:
                # e.g. GoogLeNet's aux1 and aux2 layers
                continue
            features["_".join(prefix + [name])] = ModuleHook(layer)
            hook_layers(layer, prefix=prefix + [name])

# %%
""" Main """
# hook model 
model.to(device).eval()
features = OrderedDict()
hook_layers(model)

print(f"Hooked layers:")
for k, v in features.items():
    print(k)

#%%
# get data
images, labels = next(iter(data_loader))
images = images.to(device)
labels = labels.to(device)

# forward
model(images)

#%%
# extract features you want
obj = 'layer4_2_relu3'
feat = features[obj].features

# %%
