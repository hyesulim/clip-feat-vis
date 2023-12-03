import os
import wandb
import numpy as np

import torch
import pdb

from src.models.eval import evaluate
from src.models.ft_loss import finetune
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        # return {
        #     key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        #     for key in theta_0.keys()
        # }
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_1.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_1.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        # assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def wise_ft(args):
    if args.wb_project:
        wandb_args = {'project': args.wb_project,'entity':'changdaeoh'}
        wandb_args['name'] = args.method if args.method else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)
    assert args.save is not None, 'Please provide a path to store models'
    
    # if args.load is None:
    #     # Build and save zero-shot model
    image_encoder = CLIPEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder.model)
    delattr(image_encoder.model, 'transformer')
    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    classifier.save(zeroshot_checkpoint)

    #     # Standard fine-tuning
    #     args.load = zeroshot_checkpoint
    #     args.save = os.path.join(args.save, 'finetuned')
    #     finetuned_checkpoint = finetune(args)
    # elif len(args.load) == 1:
        # Build and save zero-shot model
        
    #image_encoder = CLIPEncoder(args, keep_lang=True)
    #classification_head = get_zeroshot_classifier(args, image_encoder.model)
    #delattr(image_encoder.model, 'transformer')
    #classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    if args.bibimbap:
        zeroshot_checkpoint = os.path.join('/data1/changdae/calib-ft/checkpoints/ImageNet/ratatouille/_BS512_run1/checkpoint_0.33_0.33_0.33.pt')
    # else:
    #     zeroshot_checkpoint = os.path.join('/data1/changdae/calib-ft/checkpoints/ImageNet/zeroshot.pt')
    
    #classifier.save(zeroshot_checkpoint)
    # else:
    #     # No need to compute things from stratch
    #     assert len(args.load) == 2
    #     zeroshot_checkpoint, finetuned_checkpoint = args.load
    finetuned_checkpoint = args.clip_load
    # Load models
    
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    # delattr(zeroshot.model, 'transformer')
    # delattr(finetuned.model, 'transformer')

    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    # zsk = set(theta_0.keys())
    # ftk = set(theta_1.keys())
    # pdb.set_trace()

    # assert set(theta_0.keys()) == set(theta_1.keys())
    

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save, f'wise_ft_{args.ls}_alpha={alpha:.3f}.pt'))

        # evaluate
        # evaluate(finetuned, args)
        epoch = 0
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        args.current_epoch = epoch
        
        eval_results = evaluate(finetuned, args, train_stats=epoch_stats)

        ood_acc, num_datasets, ood_ece = 0, 0, 0.0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy': continue
                ood_acc += v
            
            if 'ECE' in k:
                if k == 'ImageNet ECE': continue
                ood_ece += v
            num_datasets += 1/2
            
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
            ood_ece = ood_ece / num_datasets
        else:
            ood_acc, ood_ece = 0, 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        epoch_stats['Avg OOD ECE'] = round(ood_ece, 4)
        wandb.log({k:v for k, v in epoch_stats.items()})

if __name__ == '__main__':
    args = parse_arguments()
    
    wise_ft(args)