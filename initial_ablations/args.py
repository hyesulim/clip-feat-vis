import os
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=1, 
        help="fixed random seed for reproduction.",
    )
    parser.add_argument("--cppn", default=False, action="store_true",
        help="cppn flag"
    )
    parser.add_argument("--fourier_basis", default=False, action="store_true",
        help="fourier space optimization flag"
    )
    parser.add_argument("--c_decorr", default=False, action="store_true",
        help="channel decorrelation"
    )
    parser.add_argument("--optim", type=str, default='adam', 
        help="optimizer name",
    )
    parser.add_argument("--lr", type=float, default=5e-2, 
        help="learning rate of optimization",
    )
    parser.add_argument("--iters", type=int, default=512,
        help="number of iterations for optimization",
    )
    parser.add_argument("--obj", type=str, default="layer4_2_relu3:486",
        help="objective for visualization",
    )
    parser.add_argument("--backbone", type=str, default="CLIP-RN50",
        help="backbone CNN such as CLIP-RN50",
    )
    parser.add_argument("--ckpt_path", type=str, default="",
        help="path of pre-trained model checkpoint",
    )
    parser.add_argument("--tfm", type=str, default="pad;jitter;rscale;rotate",
        help="transform for invariance. separated by semicolon",
    )
    parser.add_argument("--save", type=str, default="",
        help="path for saving result images",
    )
    parser.add_argument("--filename", type=str, default="",
        help="filename of result visualization",
    )

    parsed_args = parser.parse_args()
    # if parsed_args.obj.split(':')[0] not in ['layer4_2_relu3','layer4_2_conv3','layer4_2_bn3']:
    #     raise ValueError('invalid layer specficiation')


    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return parsed_args