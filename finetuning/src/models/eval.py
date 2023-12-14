import os
import json

import torch
import numpy as np
from finetuning.src.models import utils
from finetuning.src.datasets_.common import get_dataloader, maybe_dictionarize
import finetuning.src.datasets_ as datasets
import torch.nn.functional as F

from torchmetrics.classification import MulticlassCalibrationError
import pdb

import finetuning.src.visualize as visualize
import matplotlib.pyplot as plt


def eval_single_dataset(image_classifier, dataset, args, classification_head, visualization=False, dataset_name=None):
    model = image_classifier
    input_key = 'images'
    image_enc = None
    model.eval()

    if classification_head is None:
        pass
    else:
        classification_head.eval()

    dataloader = get_dataloader(dataset,
                                is_train=False,
                                args=args,
                                image_encoder=image_enc)

    batched_data = enumerate(dataloader)
    device = args.device

    #pdb.set_trace()

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        tot_ece = 0.
        
        ys, y_hats, confidences = torch.Tensor([]).to(device), torch.Tensor([]).to(device),torch.Tensor([]).to(device)

        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            ece_metric = MulticlassCalibrationError(num_classes=logits.shape[1], n_bins=10, norm='l1')

            # pdb.set_trace()
            if args.temperature_scale > 0:
                logits = logits * args.temperature_scale

            # for reliabiltiy diagram
            prob = F.softmax(logits, dim=1)
            confidence, y_hat = torch.max(prob, axis=1)

            confidences = torch.cat((confidences, confidence))
            y_hats = torch.cat((y_hats, y_hat))
            ys = torch.cat((ys, y))

            tot_ece += ece_metric(prob, y) * logits.shape[0]

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n
        mean_ece = tot_ece / n

        # args.method args.tran_dataset , args.model, args.distil_coef, args.distil_smoothing -> title?
        if visualization:
            plot_dir = './plots'
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            visualize.draw_reliability_diagram(ys.cpu(), y_hats.cpu(), confidences.cpu(), num_bins=10, title=f'{args.model}', ece=mean_ece)
            file_name = f'{dataset_name}_{args.method}_ls{args.ls}_ts{args.temperature_scale}.png'
            plt.savefig(os.path.join(plot_dir, file_name))


        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    metrics['ece'] = mean_ece.item()
    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args,
                              classification_head, data):

    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()
    device = args.device

    # pdb.set_trace()

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.
        tot_ece = 0.

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        assert x.shape[0] == 2 * args.k, 'val mismatch size'

        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        ece_metric = MulticlassCalibrationError(num_classes=logits.shape[1], n_bins=10, norm='l1')
        if args.temperature_scale > 0:
            logits = logits * args.temperature_scale
        tot_ece += ece_metric(F.softmax(logits, dim=1), y) * logits.shape[0]

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n
        mean_ece = tot_ece / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    metrics['ece'] = mean_ece.item()
    return metrics['top1'], metrics['ece'], cnt_loss.item()


def evaluate(image_classifier,
             args,
             classification_head=None,
             train_stats={},
             logger=None,
             bibim=False):
    if args.eval_datasets is None:
        return
    info = vars(args)
    
    for i, dataset_name in enumerate(args.eval_datasets):
        if bibim:
            if (i != 0) and (i != 4):
                continue

        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)

        if classification_head is None:
            dataset = dataset_class(image_classifier.val_preprocess,
                                location=args.data_location,
                                batch_size=args.batch_size)
        else:
            dataset = dataset_class(image_classifier.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.batch_size)
        
        vis_flag = True if args.vis_calibration else False
        results = eval_single_dataset(image_classifier, dataset, args,
                                      classification_head, visualization=vis_flag, dataset_name=dataset_name)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            if logger != None:
                logger.info(
                    f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)
        
        if 'ece' in results:
            print(f"{dataset_name} ECE: {results['ece']:.4f}")
            if logger != None:
                logger.info(
                    f"{dataset_name} ECE: {results['ece']:.4f}")
            train_stats[dataset_name + " ECE"] = round(results['ece'], 4)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
                if logger != None:
                    logger.info(f"{dataset_name} {key}: {val:.4f}")
                train_stats[dataset_name + key] = round(val, 4)

    return info