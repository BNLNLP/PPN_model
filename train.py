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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initiate model
model = BarValueExtractor()
model = nn.DataParallel(model)
model = model.to(device)

mlp_lr = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=mlp_lr, momentum = 0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# define print-to-console rate
num_updates_per_epoch = 5
update_steps = len(train_aug_dataloader.dataset) // (num_updates_per_epoch * bs)
#update_steps = 1

start_epoch = 1600
num_epochs = 1700

# detection and eval/error thresholds
pnt_detect_thresh = 0.9
cls_conf_thresh = 0.75
eval_thresh = 1.5/56

if start_epoch:
    # load checkpoint
    load_checkpoint_name = f'ppn_chk_epoch_{start_epoch:04}.pth'
    load_checkpoint_path = os.path.join(checkpoint_dir, load_checkpoint_name)
    checkpoint = torch.load(load_checkpoint_path)
    model.module.load_state_dict(checkpoint['model_state_dict'])

# train losses
origin_losses = []
#orient_losses = []
pclass_losses = []
pntreg_losses = []
losses = []
# test/validation losses
vorigin_losses = []
#vorient_losses = []
vpclass_losses = []
vpntreg_losses = []
vlosses = []
# training time per epoch
train_times = []
vtimes = []

barPs, barRs, barF1s, tickPs, tickRs, tickF1s, meanF1s = [], [], [], [], [], [], []

# class weights: [None, bar, tick]
class_weights = torch.tensor([0.05, 1., 1.]).to(device)

print('start training')
for epoch in range(start_epoch, num_epochs):
    t = time.time()    
    model.train()
    
    epoch_origin_loss = 0.
    epoch_pclass_loss = 0.
    epoch_pntreg_loss = 0.
    epoch_total_loss = 0.
    
    for i, (img_path, img, targets) in enumerate(train_z_dataloader):
        gt_orient, gt_origin, gt_cls_map, gt_reg_map, _, _ = targets
                
        #img = img.to(device)
        gt_orient = gt_orient.to(device)
        gt_origin = gt_origin.to(device)
        gt_cls_map = gt_cls_map.to(device)
        gt_reg_map = gt_reg_map.to(device)
        
        if epoch % 100 == 0 and i < 2:
#            print('Checking model forward time on batch...', end='')
            mt = time.time()
        orient_pred, origin_pred, bars, ticks, pts_cls_pred, pts_reg_pred = model(img)
        if epoch % 100 == 0 and i < 2:
#            print(f' done... ({time.time() - mt:0.1f} seconds)')
            pass
            
        ## COMPUTE LOSS        
        origin_loss = F.smooth_l1_loss(origin_pred, gt_origin)
        #orient_loss = F.binary_cross_entropy(orient_pred, gt_orient)
        pts_cls_loss = F.cross_entropy(pts_cls_pred, gt_cls_map, weight=class_weights, reduction='mean')
        
        # only count reg loss for cells with objects
#        positive_reg_pred = torch.where(gt_reg_map>0.0, pts_reg_pred, gt_reg_map)
#        pts_reg_loss = F.smooth_l1_loss(positive_reg_pred, gt_reg_map)
        #pts_reg_loss = F.smooth_l1_loss(pts_reg_pred, gt_reg_map)
        
        # version 2 of reg loss masking:
        gt_reg_list = torch.masked_select(gt_reg_map.permute(1,0,2,3), gt_cls_map.gt(0))
        pred_reg_list = torch.masked_select(pts_reg_pred.permute(1,0,2,3), gt_cls_map.gt(0))
        pts_reg_loss = F.mse_loss(pred_reg_list, gt_reg_list)
        
        # 2021-11-06 add actual point loss (with nms)
        gt_bars, gt_ticks = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)
        pred_bars, pred_ticks = get_pred_bars_ticks(pts_cls_pred, pts_reg_pred,
#                                                    pt_thresh = 0.99, conf_thresh = 0.7)
                                                    pnt_detect_thresh, cls_conf_thresh)
        pred_bars, pred_ticks = nms(pred_bars, pred_ticks)
        pts_list_loss = 0.
        tick_align_loss = 0.
        for bim in range(len(pred_bars)):                         # per batch image
            gbars, gticks = gt_bars[bim], gt_ticks[bim]
            pbars, pticks = pred_bars[bim], pred_ticks[bim]
            pts_list_loss += evaluate_pts_err(gbars, gticks, pbars, pticks, eval_thresh)
            tick_align_loss += sum([abs(t[0] - origin_pred[bim, 0]) for t in pticks])
        
        
        optimizer.zero_grad()
        
        loss = origin_loss + pts_cls_loss + pts_reg_loss + \
               pts_list_loss / 10. + tick_align_loss / 1000.
        
        #orient_losses.append(orient_loss.item())
        epoch_origin_loss += origin_loss.item()
        epoch_pclass_loss += pts_cls_loss.item()
        epoch_pntreg_loss += pts_reg_loss.item()
        epoch_total_loss += loss.item()

        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % update_steps == 0:
            print(f'Ep {epoch+1}, b {i+1}, L\'s: ' +
                  #'orient: {orient_loss:.4f}, ' +
                  f'orig: {origin_loss:.3f}, cls: {pts_cls_loss:.3f}, ' + 
                  f'reg: {pts_reg_loss:.3f}, pts: {pts_list_loss:.3f}, ' +
                  f'algn: {tick_align_loss:.3f}, tot: {loss:.3f}')
    
    origin_losses.append(epoch_origin_loss)
    pclass_losses.append(epoch_pclass_loss)
    pntreg_losses.append(epoch_pntreg_loss)
    losses.append(epoch_total_loss)
    
    ttime = time.time() - t
    print(f'trained epoch {epoch+1} in {ttime:0.1f} seconds. Total loss: {epoch_total_loss:.4f}')
    train_times.append(ttime)
    
    # reduce learning rate
#    if (epoch + 1) % 200 == 0:
#        mlp_lr = mlp_lr / 2
#        optimizer = torch.optim.SGD(model.parameters(), lr=mlp_lr, momentum = 0.9)
    
    epoch_vorigin_loss = 0.
    epoch_vpclass_loss = 0.
    epoch_vpntreg_loss = 0.
    epoch_vtotal_loss = 0.
    
    epoch_barP, epoch_barR, epoch_tickP, epoch_tickR = [], [], [], []
    
    # evaluate model loss on val set
    t = time.time()
    model.eval()
    for i, (img_path, img, targets) in enumerate(test_z_dataloader):
        with torch.no_grad():
            gt_orient, gt_origin, gt_cls_map, gt_reg_map, gt_bars, gt_ticks = targets
            gt_orient = gt_orient.to(device)
            gt_origin = gt_origin.to(device)
            gt_cls_map = gt_cls_map.to(device)
            gt_reg_map = gt_reg_map.to(device)        
            orient_pred, origin_pred, bars, ticks, pts_cls_pred, pts_reg_pred = model(img)
            
            origin_loss = F.smooth_l1_loss(origin_pred, gt_origin)
            pts_cls_loss = F.cross_entropy(pts_cls_pred, gt_cls_map, weight=class_weights)
            # masked reg loss
            gt_reg_list = torch.masked_select(gt_reg_map.permute(1,0,2,3), gt_cls_map.gt(0))
            pred_reg_list = torch.masked_select(pts_reg_pred.permute(1,0,2,3), gt_cls_map.gt(0))
            pts_reg_loss = F.mse_loss(pred_reg_list, gt_reg_list)
            
            # add point list distance loss
            gt_bars, gt_ticks = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)
            pred_bars, pred_ticks = get_pred_bars_ticks(pts_cls_pred, pts_reg_pred,
#                                                        pt_thresh = 0.99, conf_thresh = 0.7)
                                                        pnt_detect_thresh, cls_conf_thresh)
            pred_bars, pred_ticks = nms(pred_bars, pred_ticks)
            pts_list_loss = 0.
            tick_align_loss = 0.
            for bim in range(len(pred_bars)):                         # per batch image
                gbars, gticks = gt_bars[bim], gt_ticks[bim]
                pbars, pticks = pred_bars[bim], pred_ticks[bim]
                pts_list_loss += evaluate_pts_err(gbars, gticks, pbars, pticks, eval_thresh)
                tick_align_loss += sum([abs(t[0] - origin_pred[bim, 0]) for t in pticks])
            
            vloss = origin_loss + pts_cls_loss + pts_reg_loss + \
                    pts_list_loss / 10. + tick_align_loss / 1000.

            #vorient_losses.append(orient_loss.item())
            epoch_vorigin_loss += origin_loss.item()
            epoch_vpclass_loss += pts_cls_loss.item()
            epoch_vpntreg_loss += pts_reg_loss.item()
            epoch_vtotal_loss += vloss.item()
            
            if (epoch + 1) % 1 == 0:
                # evaluate precision, recall and F1 for bar and tick point detection
                
#                gt_bars, gt_ticks = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)
#                pred_bars, pred_ticks = get_pred_bars_ticks(pts_cls_pred, pts_reg_pred,
#                                                            pt_thresh = 0.99, conf_thresh = 0.7)
#                pred_bars, pred_ticks = nms(pred_bars, pred_ticks)
    #            print(i, len(gt_bars), len(gt_ticks), len(pred_bars), len(pred_ticks))
                for bim in range(len(pred_bars)):                         # batch image
    #                print('\t', bim)
                    gbars, gticks = gt_bars[bim], gt_ticks[bim]
                    pbars, pticks = pred_bars[bim], pred_ticks[bim]
                    barP, barR, tickP, tickR = evaluate_pts(gbars, gticks, pbars, pticks, eval_thresh)
                    epoch_barP.append(barP)
                    epoch_barR.append(barR)
                    epoch_tickP.append(tickP)
                    epoch_tickR.append(tickR)
    
    vorigin_losses.append(epoch_vorigin_loss)
    vpclass_losses.append(epoch_vpclass_loss)
    vpntreg_losses.append(epoch_vpntreg_loss)
    vlosses.append(epoch_vtotal_loss)
    
    if (epoch + 1) % 1 == 0:
        mean_barP = sum(epoch_barP) / len(epoch_barP)
        mean_barR = sum(epoch_barR) / len(epoch_barR)
        mean_tickP = sum(epoch_tickP) / len(epoch_tickP)
        mean_tickR = sum(epoch_tickR) / len(epoch_tickR)
        barPs.append(mean_barP)
        barRs.append(mean_barR)
        barF1s.append(f1(mean_barP, mean_barR))
        tickPs.append(mean_tickP)
        tickRs.append(mean_tickR)
        tickF1s.append(f1(mean_tickP, mean_tickR))
        meanF1s.append((barF1s[-1] + tickF1s[-1]) / 2.)
    
    vtime = time.time() - t
    print (f'ev: {vtime:0.1f}s; VL: {epoch_vtotal_loss:.3f}; ', end='')
    if (epoch + 1) % 1 == 0:
        print (f'bP: {mean_barP:.3f}, bR: {mean_barR:.3f}, tP: {mean_tickP:.3f}, ' +
               f'tR: {mean_tickR:.3f}, bF1: {barF1s[-1]:.4f}, tF1: {tickF1s[-1]:.4f}, ' +
               f'avgF1: {meanF1s[-1]:.4f}')
    else:
        print('')
    vtimes.append(vtime)

    # save model
    if (epoch + 1) % 20 == 0:
        checkpoint_name = f'ppn_chk_epoch_{epoch+1:04}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, checkpoint_path)

# log losses and training times to file
log_losses(start_epoch, num_epochs,
           origin_losses, pclass_losses, pntreg_losses, losses, train_times,
           vorigin_losses, vpclass_losses, vpntreg_losses, vlosses, vtimes,
           barPs, barRs, barF1s, tickPs, tickRs, tickF1s, meanF1s)