import os

import torch
import pickle
from tqdm import tqdm
import math

import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


# def linear_schedule(base_value, final_value, cur_iter, tot_iter):
#     '''
#     - cur_iter : start from 0
#     '''
#     schedule = ((final_value - base_value) / tot_iter) * cur_iter + base_value
#     return schedule


def accuracy(output, target, topk=(1, )):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0,
                                                  keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier, classification_head=None):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
        if classification_head is None:
            return classifier(inputs)
        classification_head = classification_head.to(inputs.device)
    feats = classifier(inputs)
    return classification_head(feats)


def get_feats(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    feats = classifier(inputs)
    # feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()