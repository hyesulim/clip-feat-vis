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
from src.args import parse_arguments
import logging
import random

import wandb
import glob
import copy

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    np.random.seed(SEED)
    random.seed(SEED)

def get_lambda_set(num=20):
    _tmp = []
    # _tmp.append([1.0, 0.0, 0.0])
    # _tmp.append([0.0, 1.0, 0.0])
    # _tmp.append([0.0, 0.0, 1.0])
    
    for i in range(num):
        _tmp.append(np.random.dirichlet((1,1,1)))
    return _tmp

def average_state_dict(model_list, ratio):    
    avg_state_dict = {}
    for key in model_list[0].keys():
        for idx, model in enumerate(model_list):
            if idx == 0:
                avg_state_dict[key] = model[key]*ratio[0]
            else:
                avg_state_dict[key] += model[key]*ratio[idx]
    return avg_state_dict

def main(args):
    set_seed(args.run)
    if args.wb_project:
        wandb_args = {'project': args.wb_project,'entity':'changdaeoh'}
        wandb_args['name'] = args.method if args.method else None
        wandb.init(**wandb_args, config=vars(args), save_code=False)
        # for path in glob.glob('**/*.py', recursive=True):
        #     wandb.save(path)

    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + '_wd' +str(args.wd) + '_lr' +str(args.lr) +"_bbsam"+str(args.bb_diri) + "_run" + str(args.run)

    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + '_wd' +str(args.wd) + '_lr' +str(args.lr) +"_bbsam"+str(args.bb_diri) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert args.save is not None, 'Please provide a path to store models'
    #############################################################################

    logger.info(args)
    
    if args.temperature_scale > 0:
        epoch_stats = {'epoch':0}
        image_clf = ImageClassifier.load(args.clip_load)
        _ = evaluate(image_clf, args, train_stats=epoch_stats, bibim=False)

        ood_acc = 0
        num_datasets = 0
        ood_ece = 0.0

        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy':
                    #ignore the ID acc term
                    continue
                ood_acc += v

            if 'ECE' in k:
                if k == 'ImageNet ECE':
                    continue
                ood_ece += v
            num_datasets += 1/2
            
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
            ood_ece = ood_ece / num_datasets
        else:
            ood_acc = 0
            ood_ece = 0

        # epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        # print(f"Avg OOD Acc : {ood_acc:.4f}")
        # epoch_stats['Avg OOD ECE'] = round(ood_ece, 4)
        # print(f"Avg OOD ECE : {ood_ece:.4f}")
        wandb.log({k:v for k, v in epoch_stats.items()})


    else:
        sdict_list = []
        for m_path in args.ratatouille_pathes.split(';'):
            image_clf = ImageClassifier.load(m_path)
            sdict_list.append(image_clf.state_dict())
        
        #* need sanity check
        if args.bibimbap:
            if args.bb_diri == 0:
                ens_coef_list = [[1/3, 1/3, 1/3]]
            else:
                ens_coef_list = get_lambda_set(args.bb_diri)
                #original_eval_ds = args.eval_datasets
                #diri_eval_ds = original_eval_ds[0]
                #print('diri eval on:', diri_eval_ds)
                #args.eval_datasets = diri_eval_ds
        else:
            ens_coef_list = [[1/3, 1/3, 1/3]]

        if args.best_ens_path:
            pass
        else:
            best_id_acc, best_ood_acc, best_model_dict, best_coef = 0, 0, None, []
            for ens_coef in ens_coef_list:
                ens_dict = average_state_dict(sdict_list, ratio=ens_coef)
                image_clf.load_state_dict(ens_dict)
                
                if args.bb_diri != 0:
                    epoch_stats = {'epoch':0}
                    # Saving model
                    print('ens coef:',ens_coef)
                    _ = evaluate(image_clf, args, train_stats=epoch_stats, bibim=True)
                    id_acc = epoch_stats['ImageNet Accuracy']
                    ood_acc = epoch_stats['ImageNetSketch Accuracy']
                    if id_acc > best_id_acc:
                        best_id_acc = id_acc
                        best_ood_acc = ood_acc
                        best_coef = ens_coef
                        best_model_dict = copy.deepcopy(ens_dict)

        if args.bb_diri != 0:
            if args.best_ens_path:
                print('load best ens ckpt')
                image_clf = image_clf.load(args.best_ens_path)
            else:
                print('=================================')
                print(f'best ID Acc: {best_id_acc:.3f}, Corresponding OOD Acc: {best_ood_acc:.3f}')
                print('=================================')
                image_clf.load_state_dict(best_model_dict)
                image_clf.save(args.save + f'/{best_coef[0]:.2f}_{best_coef[1]:.2f}_{best_coef[2]:.2f}.pt')
        
        if args.bibimbap:
            #! further fine-tuning !!
            # if args.bb_diri != 0:
            #     args.eval_datasets = original_eval_ds
            finetuned_checkpoint = finetune(args, image_clf)
        else:
            epoch_stats = {'epoch':0}
            _ = evaluate(image_clf, args, train_stats=epoch_stats)

            ood_acc = 0
            num_datasets = 0
            ood_ece = 0.0

            for k, v in epoch_stats.items():
                if 'Accuracy' in k:
                    if k == 'ImageNet Accuracy':
                        #ignore the ID acc term
                        continue
                    ood_acc += v

                if 'ECE' in k:
                    if k == 'ImageNet ECE':
                        continue
                    ood_ece += v
                num_datasets += 1/2
                
            if num_datasets != 0:
                ood_acc = ood_acc / num_datasets
                ood_ece = ood_ece / num_datasets
            else:
                ood_acc = 0
                ood_ece = 0

            epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
            print(f"Avg OOD Acc : {ood_acc:.4f}")
            epoch_stats['Avg OOD ECE'] = round(ood_ece, 4)
            print(f"Avg OOD ECE : {ood_ece:.4f}")
            wandb.log({k:v for k, v in epoch_stats.items()})

        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            if args.best_ens_path:
                coef_subfix = '0.19_0.62_0.19'
            else:
                coef_subfix = '_'.join([str(coef)[:4] for coef in best_coef])
            model_path = os.path.join(args.save, f'checkpoint_{args.epochs}_{coef_subfix}.pt')
            print('Saving model to', model_path)
            image_clf.save(model_path)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
