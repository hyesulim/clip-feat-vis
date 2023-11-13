# %%
import sys
sys.path.append("../")

import torch
import torch.nn as nn

import clip
from linear_probe.dataset import get_combined_loader
from linear_probe.helpers import plot_images

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
""" (Optional) Visualize data """
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

print("Hooked layers:")
for k, v in features.items():
    print(k)

# %%
# get data
images, labels = next(iter(data_loader))
images = images.to(device)
labels = labels.to(device)

# forward
model(images)

# %%
# extract features you want
obj = "layer1_2_relu3"
feat = features[obj].features

# %%
print("Hooked layers:")
for k, v in features.items():
    try:
        print(k, v.features.size())
    except:
        print(k)
