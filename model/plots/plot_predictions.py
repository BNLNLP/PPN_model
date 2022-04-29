# Carlos X. Soto, csoto@bnl.gov, 2022

import os
import sys
import time
from datetime import datetime
from ../BarValueExtractor import BarValueExtractor
from ../nms import nms
import ../dataloaders
import ../utils/metrics
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

### PLOT predicted points on source images

epoch = '1640'
#epoch = '9800'
checkpoint_dir = 'ppn_checkpoints/zhao'
#checkpoint_dir = 'ppn_checkpoints/20211110_annot_10k_lr1_noalignloss'
#checkpoint_dir = 'ppn_checkpoints/sep30.2021_add_lrdecay'
checkpoint_name = f'ppn_chk_epoch_{epoch}.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

model = BarValueExtractor()#.to(device)
#model = nn.DataParallel(model)
#model = model.to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

timestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_trainset'
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_testset_nolrdecay'
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_bioimgs_nolrdecay'
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_testset2'
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_testset3'
#out_im_dir = f'ppn_out/{timestring}_ck{epoch}_testset4'
out_im_dir = f'ppn_out/{timestring}_ck{epoch}_zhou_test'

save_outimgs = True
if save_outimgs:
    if not os.path.isdir(out_im_dir):
        os.mkdir(out_im_dir)

model.eval()

#max_output = 1000
max_output = 64

num_output = 0
tstart = time.time()

# rectangle/ellipse width/radius
r = 8

# thresholds
pnt_detect_thresh = 0.9
cls_conf_thresh = 0.75
eval_thresh = 1.5 / 56

barF1s, tickF1s = [], []

#for k, (img_path, img, targets) in enumerate(train_dataloader):
#for k, (img_path, img, targets) in enumerate(test_dataloader):
#for k, (img_path, img) in enumerate(bioimg_dataloader):
for k, (img_path, img, targets) in enumerate(test_z_dataloader):
#for k, (img_path, img, targets) in enumerate(test4_dataloader):
    
    gt_orient, gt_origin, gt_cls_map, gt_reg_map, _, _ = targets
    gt_bars, gt_ticks = pts_map_to_lists_v2(gt_cls_map, gt_reg_map)
#    print('pts_map_to_lists_v2 bars', gt_bars[0])
#    break
    
    _, origin_pred, _, _, pred_cls_map, pred_reg_map = model(img)
    
#    bars, ticks = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pt_thresh = 0.99, conf_thresh = 0.7)
    bars, ticks = get_pred_bars_ticks(pred_cls_map, pred_reg_map, pnt_detect_thresh, cls_conf_thresh)
    bars, ticks = nms(bars, ticks)
    
    for i, path in enumerate(img_path):
        draw_img = Image.open(path)
        im_dimx, im_dimy = draw_img.size
        #print(im_dimx, im_dimy)
        
        draw = ImageDraw.Draw(draw_img)
        
        # re-compute image padding
        w, h = draw_img.size
        dim_diff = abs(h - w)
        padded_wh = max(w, h)                                   # padded to square
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # (left, right, top, bottom)
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        
        # origin and bar/tick points are [0-1], relative to padded_wh
        
        #o_x = gt_origin[i][0].item() * im_dimx
        #o_y = gt_origin[i][1].item() * im_dimy
#        o_x = origin_pred[i][0].item() * im_dimx
#        o_y = origin_pred[i][1].item() * im_dimy
        o_x = origin_pred[i][0].item() * padded_wh - pad[0]
        o_y = origin_pred[i][1].item() * padded_wh - pad[2]
        #draw.rectangle([o_x-3, o_y-3, o_x+3, o_y+3], fill = 'white', outline='green', width=4)
#        draw.ellipse([o_x-r, o_y-r, o_x+r, o_y+r], fill = 'green', outline='green', width=4)
        
        # TODO: need to keep track of padding, or else not apply padding and instead
        # stretch image when resizing...
        
        #for b in gt_bars[i]:
        for b in bars[i]:
#            b_x = b[0].item() * im_dimx
#            b_y = b[1].item() * im_dimy
            b_x = b[0].item() * padded_wh - pad[0]
            b_y = b[1].item() * padded_wh - pad[2]
            #draw.rectangle([b_x-3, b_y-3, b_x+3, b_y+3], fill = 'white', outline='red', width=4)
            draw.ellipse([b_x-r, b_y-r, b_x+r, b_y+r], fill = 'red', outline='red', width=4)
        
        #for t in gt_ticks[i]:
        for t in ticks[i]:
#            t_x = t[0].item() * im_dimx
#            t_y = t[1].item() * im_dimy
            t_x = t[0].item() * padded_wh - pad[0]
            t_y = t[1].item() * padded_wh - pad[2]
            #draw.rectangle([t_x-3, t_y-3, t_x+3, t_y+3], fill = 'white', outline='blue', width=4)
            draw.ellipse([t_x-r, t_y-r, t_x+r, t_y+r], fill = 'blue', outline='blue', width=4)
        
        # redraw origin
#        draw.ellipse([o_x-r, o_y-r, o_x+r, o_y+r], fill = 'green', outline='green', width=4)
#        print("origin:", origin_pred[i][0].item(), origin_pred[i][1].item())
        
        # EVALUATION
        barP, barR, tickP, tickR = evaluate_pts(gt_bars[i], gt_ticks[i], bars[i], ticks[i], eval_thresh)
        barF1s.append(f1(barP, barR))
        tickF1s.append(f1(tickP, tickR))
        
        out_filename = os.path.join(out_im_dir, path.split('/')[-1])
        print(f'{num_output: 4}\t{out_filename}', end='')
        print(f'\t{len(bars[i])} bars, {len(ticks[i])} ticks')
        print(f'\tbP: {barP:.4f}, bR: {barR:.4f}, tP: {tickP:.4f}, tR: {tickR:.4f}, ' +
              f'bF1: {barF1s[-1]:.4f}, tF1: {tickF1s[-1]:.4f}')
        
        if save_outimgs:
            out_filename = os.path.join(out_im_dir, path.split('/')[-1])
            draw_img.save(out_filename)

            # save class maps (gt & pred), (model img needs value scaling to 0-1 or 0-255)
            out_filename = os.path.join(out_im_dir, 'pointmap_' + path.split('/')[-1])
            plt.imsave(out_filename, F.sigmoid(pred_cls_map.detach()[i,0] * -1.))
            out_filename = os.path.join(out_im_dir, 'barmap_' + path.split('/')[-1])
            plt.imsave(out_filename, F.sigmoid(pred_cls_map.detach()[i,1]))
            out_filename = os.path.join(out_im_dir, 'tickmap_' + path.split('/')[-1])
            plt.imsave(out_filename, F.sigmoid(pred_cls_map.detach()[i,2]))
        
#        out_filename = os.path.join(out_im_dir, 'gtmap_' + path.split('/')[-1])
#        plt.imsave(out_filename, gt_cls_map[i])
        
        num_output += 1
        if num_output >= max_output:
            break
    
    if num_output >= max_output:
        break
print(f'inference time for {num_output} images: {time.time() - tstart:.1f} seconds')

mean_bF1 = sum(barF1s) / len(barF1s)
mean_tF1 = sum(tickF1s) / len(tickF1s)
meanF1 = (mean_bF1 + mean_tF1) / 2.
print(f'mean bar F1: {mean_bF1:.4f}, mean tick F1: {mean_tF1:.4f}')