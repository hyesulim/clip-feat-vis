import os

import torch
import clip

from lucent.optvis import render, param
from args import parse_arguments
from utils import set_seed, set_transform, set_optimizer
import pdb
from PIL import Image as im 

from datetime import datetime


def main(args):
    set_seed(args.seed)

    model_flag, backbone = args.backbone.split('-')
    if model_flag == 'CLIP' :
        model, preprocess = clip.load(backbone, device=args.device)
        if args.ckpt_path:
            model.load_state_dict(torch.load(args.ckpt_path))
        model = model.visual
    else:
        raise ValueError('not implemented yet')

    model.to(args.device).eval()

    if args.cppn:
        param_f = lambda: param.cppn(224)
    else:
        param_f = lambda: param.image(224, fft=args.fourier_basis, decorrelate=args.c_decorr)
    
    opt = set_optimizer(args.optim, args.lr)
    transforms = set_transform(args.tfm)

    out = render.render_vis(model, 
                            args.obj, 
                            param_f, 
                            opt, 
                            transforms=transforms, 
                            thresholds=(args.iters,),
                            random_seed=args.seed)
    
    if args.save:
        now = datetime.now()
        import matplotlib.pyplot as plt
        os.makedirs(args.save, exist_ok=True)
        plt.imshow(out[0][0])
        plt.xticks([]); plt.yticks([])
        plt.savefig(args.save+f'/{args.filename}_{now.strftime("%Y_%m_%d_%H_%M_%S")}.png')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)