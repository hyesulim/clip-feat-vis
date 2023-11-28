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
    args,
    obj="layer1_2_relu3",
):
    # set save_dir for logging
    base_dir = os.path.join(args.root_code, "logs", args.lp_dataset, args.obj)
    save_dir = make_save_dir(base_dir)
    log_message(str(args), save_dir)

    # hook feature extractor
    feature_ext.to(args.device).eval()
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
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(linear_probe.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(linear_probe.parameters(), lr=args.lr, weight_decay=5e-5)
    else:
        raise NotImplementedError

    # train
    for epoch in range(args.num_epochs):
        loss_total = 0.0
        acc_total = 0.0
        for i, batch in enumerate(tqdm(data_loader)):
            images = batch[0].to(args.device)
            labels = batch[1].to(args.device).to(dtype=torch.float).view(-1, 1)
            # labels = torch.Tensor(labels, dtype=torch.float).view(-1, 1)

            # extract features
            with torch.no_grad():
                feature_ext(images)
                feat = features[obj].features
                feat_flatten = feat.reshape((feat.shape[0], -1))
                # feat_flatten = feat_flatten.to(dtype=torch.float32)

            # linear probing
            out = linear_probe(feat_flatten.to(dtype=torch.float))  # logits

            loss = criterion(out, labels)
            loss_total += loss.item()

            # import pdb

            # pdb.set_trace()
            # pred = out.argmax(1)
            train_acc = ((torch.sigmoid(out) > 0.5) == labels).sum() / out.shape[0]
            acc_total += train_acc

            if i % 10 == 1:
                msg = f"Epoch [{epoch+1}/{args.num_epochs}] iter {i}, Trianing loss: {loss.item():.4f}, Train acc: {train_acc:.4f}"
                print(msg)
                log_message(msg, save_dir)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg = loss_total / len(data_loader)
        acc_avg = acc_total / len(data_loader)
        msg = f"Epoch [{epoch+1}/{args.num_epochs}], Trianing loss: {loss_avg:.4f}, Train acc: {acc_avg:.4f}"
        print(msg)
        log_message(msg, save_dir)

    save_model(linear_probe, save_dir)

    return linear_probe, save_dir
