from omegaconf import OmegaConf
from tqdm import tqdm
import time
import torch
import numpy as np
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check
from configs import build_config
from utils_pel import setup_seed
from log import get_logger
from pel4vad import init_pel, infer_pel, cal_false_alarm, get_optimal_threshold
from yolov9 import init_yolov9, infer_yolov9, metrics_yolov9
from utils.torch_utils import select_device

def main(args_cli, cfg):
    st = time.time()
    
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    logger.info('Arguments:{}'.format(OmegaConf.to_yaml(args)))
    if args.on_extraction in ['save_numpy', 'save_pickle']:
        logger.info('Saving features to {}'.format(args.output_path))

    # Initialize/load model and set device
    device = select_device(args.device, batch_size=cfg.batch_size)
    logger.info('Device: {}'.format(args.device))

    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == 'i3d':
        from models.i3d.extract_i3d import ExtractI3D as Extractor
    else:
        raise NotImplementedError(f'Extractor {args.feature_type} is not implemented.')

    extractor = Extractor(args)

    model_pel, frame_gt = init_pel(cfg, device, logger)
    model_yolo, results_yolo = init_yolov9(cfg, device)
    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths, to_shuffle=False)

    logger.info('The number of specified videos: {}'.format(len(video_paths)))

    with torch.no_grad():
        model_pel.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(frame_gt.copy()).cuda()
        total_frame = 0
        total_frame_anomaly = 0
        results_pel = (pred, normal_preds, normal_labels, gt_tmp)
        for video_path in tqdm(video_paths):
            logger.info('Start Testing: {}'.format(video_path))
            feature_stack, frame_stack = extractor._extract(video_path)  # note the `_` in the method name
            anomaly_scores, results_pel = infer_pel(logger, cfg, model_pel, feature_stack, results_pel)
            frame_stack = frame_stack[:len(anomaly_scores)]   #same shape
            logger.info('This video number of frames: {}, number of anomaly scores: {}'.format(len(frame_stack), len(anomaly_scores)))
            
            if np.sum(anomaly_scores) > 0.0:
                logger.info('Test yolov9')
                results_yolo = infer_yolov9(cfg, model_yolo, results_yolo, frame_stack, video_path, anomaly_scores)
            total_frame += len(frame_stack)
            total_frame_anomaly += np.sum(anomaly_scores)

        pred, normal_preds, normal_labels, gt_tmp = results_pel
        logger.info('Total number of frames: {}, total number of anomaly frames: {}'.format(total_frame, total_frame_anomaly))
    
        pred = list(pred.cpu().detach().numpy())
        far = cal_false_alarm(logger, normal_labels, normal_preds)
        fpr, tpr, thresholds1 = roc_curve(list(frame_gt), np.repeat(pred, 16))
        
        roc_auc = auc(fpr, tpr)
        pre, rec, thresholds2 = precision_recall_curve(list(frame_gt), np.repeat(pred, 16))

        fscore = (2 * pre * rec) / (pre + rec)

        get_optimal_threshold(logger, tpr, fpr, fscore, thresholds1, thresholds2)
        
        pr_auc = auc(rec, pre)
        metrics_yolov9(cfg, model_yolo, results_yolo)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    cfg = build_config('ucf')
    logger = get_logger(cfg.logs_dir)
    main(args_cli, cfg)