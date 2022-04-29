#! /usr/bin/env python
# Carlos X. Soto, csoto@bnl.gov, 2022


import os
import time
import math
import argparse
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BarValueExtractor import *
from model.Datasets import *
from model.dataloaders import *
from model.nms import *
from model.utils.metrics import *
from model.utils.map_conversions import *

## Evaluate performance on test sets

#checkpoint_dir = 'ppn_checkpoints/20211002_ml_origin'
#checkpoint_dir = 'ppn_checkpoints/20211107_synth_ptsloss_1500'
#checkpoint_dir = 'ppn_checkpoints/20211107_synth_align_1600'
#checkpoint_dir = 'ppn_checkpoints/20211108_annot_3k-5k'
#checkpoint_dir = 'ppn_checkpoints/20211003_synthetic'
#checkpoint_dir = 'ppn_checkpoints'
checkpoint_dir = 'ppn_checkpoints/submission_chks'

#eval_dataloader = val_aug_dataloader
#eval_dataloader = test4_dataloader
eval_dataloader = zhao_test_dl

# defaults
#epoch = 9800
epoch = 300
pnt_detect_thresh = 0.9
cls_conf_thresh = 0.75
eval_thresh = 1.5 / 56


print('epoch, det_thresh, cls_thresh, ev_thresh, time, bar P, bar R, tick P, tick R, bar F1, tick F1, mean F1')
#for epoch in range(1000,1501,20):
#for epoch in [1600,9800,300]:
#for pnt_detect_thresh in range(80,100):
#    pnt_detect_thresh = pnt_detect_thresh / 100
#for cls_conf_thresh in range(50,100,5):
#    cls_conf_thresh = cls_conf_thresh / 100
#for eval_thresh in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
for eval_thresh in [2.8, 0.5]:
#    eval_thresh = eval_thresh / 56
    
    checkpoint_name = f'ppn_chk_epoch_{epoch:04}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    model = BarValueExtractor()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tstart = time.time()
    
    img_metrics = []   # (precision, recall, F1) x2 (bar, tick)
    for k, (img_path, img, targets) in enumerate(eval_dataloader):
        gt_orient, gt_origin, gt_cls_map, gt_reg_map, _, _ = targets
        gt_bars, gt_ticks = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)

        _, origin_pred, _, _, pred_cls_map, pred_reg_map = model(img)

#        bars, ticks = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh = 0.99, conf_thresh = 0.7)
        bars, ticks = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pnt_detect_thresh, cls_conf_thresh)
        bars, ticks = nms(bars, ticks)

        # EVALUATION per image
        for i, path in enumerate(img_path):
            barP, barR, tickP, tickR = evaluate_pts(gt_bars[i], gt_ticks[i], bars[i], ticks[i], eval_thresh)
            barF1, tickF1 = f1(barP, barR), f1(tickP, tickR)
            img_metrics.append((barP, barR, barF1, tickP, tickR, tickF1))

    # mean of bar and tick metrics (precision, recall, F1) @ threshold of 1.5/56
    mbP = sum([im[0] for im in img_metrics]) / len(img_metrics)
    mbR = sum([im[1] for im in img_metrics]) / len(img_metrics)
    mbF1 = sum([im[2] for im in img_metrics]) / len(img_metrics)
    mtP = sum([im[3] for im in img_metrics]) / len(img_metrics)
    mtR = sum([im[4] for im in img_metrics]) / len(img_metrics)
    mtF1 = sum([im[5] for im in img_metrics]) / len(img_metrics)
    avgF1 = (mbF1 + mtF1) / 2.

    print(f'{epoch}, {pnt_detect_thresh:.2f}, {cls_conf_thresh:.2f}, ' +
          f'{eval_thresh:.1f}, {time.time() - tstart:.1f}, '+
          f'{mbP:.4f}, {mbR:.4f}, {mtP:.4f}, {mtR:.4f}, ' +
          f'{mbF1:.4f}, {mtF1:.4f}, {avgF1:.6f}')