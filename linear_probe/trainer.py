""" 
Linear probing
"""
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from linear_probe.helpers import *
from lucent.optvis.render import ModuleHook


def train(
    feature_ext,
    linear_probe,
    data_loader,
    config,
    obj="layer1_2_relu3",
):
    # set save_dir for logging
    base_dir = os.path.join(config["root_code"], "logs", config["obj"])
    save_dir = make_save_dir(base_dir)
    log_message(str(config), save_dir)

    # hook feature extractor
    feature_ext.to(config["device"]).eval()
    features = OrderedDict()

    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(feature_ext)

    # make linear probing model trainable
    linear_probe.train()

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if config["optim"] == "adam":
        optimizer = torch.optim.Adam(linear_probe.parameters(), lr=config["lr"])
    else:
        raise NotImplementedError

    # train
    for epoch in range(config["num_epochs"]):
        loss_total = 0.0
        for images, labels in tqdm(data_loader):
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            # extract features
            with torch.no_grad():
                feature_ext(images)
                feat = features[obj].features
                feat_flatten = feat.reshape((feat.shape[0], -1))
                # feat_flatten = feat_flatten.to(dtype=torch.float32)

            # linear probing
            out = linear_probe(feat_flatten.to(dtype=torch.float32))
            loss = criterion(out, labels)
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg = loss_total / len(data_loader)
        msg = f"Epoch [{epoch+1}/{config['num_epochs']}], Trianing loss: {loss_avg:.4f}"
        print(msg)
        log_message(msg, save_dir)

    save_model(linear_probe, save_dir)

    return linear_probe, save_dir
