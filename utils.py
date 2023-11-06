import torch
import numpy as np
import random
from lucent.optvis import transform

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    np.random.seed(SEED)
    random.seed(SEED)

def set_transform(tfm):
    all_transforms = []
    if 'pad' in tfm: all_transforms += [transform.pad(28)] # 16 * 7/4
    if 'jitter' in tfm: all_transforms += [transform.jitter(14)] # 8 * 7/4
    if 'rscale' in tfm: all_transforms += [transform.random_scale([n/100. for n in range(80, 120)])]
    if 'rotate' in tfm: all_transforms += [transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2)))]
    if 'jitter' in tfm: all_transforms += [transform.jitter(4)] # 2 * 7/4
    return all_transforms