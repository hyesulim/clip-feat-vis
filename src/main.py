from ast import arg
import os
import numpy as np
import torch
from src.models.eval import evaluate
from src.models.ft_loss import finetune
from src.models.flyp_loss import flyp_loss
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import logging
import random

import wandb
import glob

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    np.random.seed(SEED)
    random.seed(SEED)

def main(args):
    set_seed(args.run)
    if args.wb_project:
        wandb_args = {'project': args.wb_project,'entity':'changdaeoh'}
        wandb_args['name'] = args.method if args.method else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)
        # for path in glob.glob('**/*.py', recursive=True):
        #     wandb.save(path)

    #! V or R
    mod_flag = args.model[0]

    ###logging##################################################################
    
    # os.makedirs(args.save + args.exp_name, exist_ok=True)
    # args.save = args.save + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
    #     args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
    #         args.lr) + "_D" + str(args.distil_coef) +"_ms" + str(args.m_sche_src) +"_mr" + str(
    #             args.m_sche_tar) + "_mw" + str(args.m_warm_up) + "_dsch" + str(args.distil_schedule) + "_fp16" + str(
    #                 args.use_fp16)  + "_ds" + str(args.distil_smoothing)  + "emafr" + str(
    #                     args.ema_up_freq) + "_ls" + str(args.ls) + "_run" + str(args.run)

    # os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    # logging_path = "expt_logs/" + args.exp_name + "/" + f"{mod_flag}" + '_ep' + str(args.epochs) + "_BS" + str(
    #     args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
    #             args.m_sche_tar) + "_mw" + str(args.m_warm_up) + "_dsch" + str(args.distil_schedule) + "_fp16" + str(
    #                 args.use_fp16)  + "_ds" + str(args.distil_smoothing)  + "emafr" + str(
    #                     args.ema_up_freq) + "_ls" + str(args.ls) + "_run" + str(args.run)
    
    # os.makedirs(logging_path, exist_ok=True)
    # log_filename = logging_path + "/log.log"
    # logging.basicConfig(filename=log_filename,
    #                     format='%(asctime)s %(message)s',
    #                     filemode='w')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # assert args.save is not None, 'Please provide a path to store models'
    #############################################################################

    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    if args.head_path:
        load_epoch = 5 if args.train_dataset == 'ImageNet' else 10
        head_path = os.path.join(args.head_path, f'checkpoint_lp_{load_epoch}.pt')
        classification_head = ClassificationHead.load(head_path)
    else:
        if args.method in ['lp','lp_fp16']:
            outdim = 0
            if args.train_dataset == 'ImageNet': outdim = 1000
            elif args.train_dataset == 'IWildCamID': outdim = 182
            elif args.train_dataset == 'FMOWID': outdim = 62
            elif args.train_dataset == 'sst2Val': outdim = 2
            elif args.train_dataset == 'PatchCamelyonVal': outdim = 2
            elif args.train_dataset == 'Caltech101Val': outdim = 101
            elif args.train_dataset == 'StanfordCarsVal': outdim = 196
            elif args.train_dataset == 'Flowers102Val': outdim = 102
            classification_head = ClassificationHead(normalize=None, weights=None, shape=[512, outdim])
        else:
            classification_head = get_zeroshot_classifier(args, clip_encoder.model)

    #logger.info(args)

    if args.method in ['ft_origin','ft','ft_fp16','lp','lpft']:
        delattr(clip_encoder.model, 'transformer')
        image_clf = ImageClassifier(clip_encoder, classification_head, process_images=False)
        #finetuned_checkpoint = finetune(args, image_clf)
        print('load model')
        image_clf = image_clf.load('/data1/changdae/calib-ft/checkpoints/ImageNet/ft/R_BS512_WD0.1_LR3e-05_D0.0_ms0.05_mr0.9_mw0.2_dsch_fp161_ds1.0emafr1_ls0.0_run1/checkpoint_10.pt')
        print('re-save model')
        torch.save(image_clf,'/data1/changdae/11785-f23-prj/RN50_FT.pt')
        exit()
    elif args.method in ['flyp','zs','ours','ours_fp16','ours_ema','ours_ema_fp16','flyp_fp16']:
        if args.ce_ablation:
            finetuned_checkpoint = ce_ablation(args, clip_encoder,
                                                classification_head, logger)
        else:
            finetuned_checkpoint = flyp_loss(args, clip_encoder,
                                                classification_head, logger)
    else:
        raise ValueError('not implemented yet')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
