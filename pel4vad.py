import torch.optim as optim
import torch
import numpy as np
from model import XModel
from sklearn.metrics import confusion_matrix

from dataset import *
from utils_pel import fixed_smooth, slide_smooth

import copy
import os

def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')

def get_optimal_threshold(logger, tpr, fpr, fscore, thresholds1, thresholds2):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_roc = thresholds1[optimal_idx]
    
    optimal_idx = np.argmax(fscore)
    optimal_threshold_pr = thresholds2[optimal_idx]
    logger.info('Optimal threshold value. by ROC curve: {}, by P-R curve: {}'.format(optimal_threshold_roc, optimal_threshold_pr))

def cal_false_alarm(logger, gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    logger.info('Calculate false alarm rate. Total normal video frames= {}'.format(len(preds)))
    far = fp / (fp + tn)
    return far

def init_pel(cfg, device, logger):
    model = XModel(cfg)
    gt = np.load(cfg.gt)
    # device = torch.device("cuda")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    if cfg.ckpt_path is not None:
        load_checkpoint(model, cfg.ckpt_path, logger)
    else:
        logger.info('infer from random initialization')

    return model, gt

def inference(cfg, model, v_input, results):
    pred, normal_preds, normal_labels, gt_tmp = results
    v_input = torch.from_numpy(v_input).float().cuda(non_blocking=True)
    seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
    logits, _ = model(v_input, seq_len)
    logits = torch.mean(logits, 0)
    logits = logits.squeeze(dim=-1)

    seq = len(logits)
    if cfg.smooth == 'fixed':
        logits = fixed_smooth(logits, cfg.kappa)
    elif cfg.smooth == 'slide':
        logits = slide_smooth(logits, cfg.kappa)
    else:
        pass
    logits = logits[:seq]

    pred = torch.cat((pred, logits))
    labels = gt_tmp[: seq_len[0]*16]
    if torch.sum(labels) == 0:
        normal_labels = torch.cat((normal_labels, labels))
        normal_preds = torch.cat((normal_preds, logits))
    gt_tmp = gt_tmp[seq_len[0]*16:]

    output = list(logits.cpu().detach().numpy())
    output = np.repeat(output, 16)
    output[output < cfg.anomaly_threshold] = 0
    output[output >= cfg.anomaly_threshold] = 1

    return output, (pred, normal_preds, normal_labels, gt_tmp)

def infer_pel(logger, cfg, model, v_input, results):
    logger.info('Test PEL')
    
    output, results = inference(cfg, model, v_input, results)

    return output, results