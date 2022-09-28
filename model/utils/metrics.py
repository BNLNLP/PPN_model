# Carlos X. Soto, csoto@bnl.gov, 2022

import numpy as np

# evaluation function (for a single image)
def evaluate_pts(gt_bars, gt_ticks, pred_bars, pred_ticks, dist_thresh = 1.5 / 56):
    #for gb in gt_bars:
    #    for pb in pred_bars:
    #        pixel_dist = ((gb[0] - pb[0]) ** 2 + (gb[1] - pb[1]) ** 2) ** (0.5)
    
    # fill a matching matrix, check rows (ground truth) for uniqueness
    bar_matches = [[((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) if
                    ((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_bars] for gb in gt_bars]
    for i in range(len(gt_bars)):
        min_dist = min([bm for bm in bar_matches[i] if bm > 0], default = 0.)
        bar_matches[i] = [m if m <= min_dist else 0 for m in bar_matches[i]]
    
    num_matches = np.count_nonzero(bar_matches)
    
    bar_precision = (num_matches / len(pred_bars)) if len(pred_bars) != 0 else 0
    bar_recall = (num_matches / len(gt_bars)) if len(gt_bars) != 0 else 0
    
    ## again for ticks...
    tick_matches = [[((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) if
                     ((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in pred_ticks] for gt in gt_ticks]
    for i in range(len(gt_ticks)):
        min_dist = min([tm for tm in tick_matches[i] if tm > 0], default = 0.)
        tick_matches[i] = [m if m <= min_dist else 0 for m in tick_matches[i]]
    
    num_matches = np.count_nonzero(tick_matches)
    
    tick_precision = (num_matches / len(pred_ticks)) if len(pred_ticks) != 0 else 0
    tick_recall = (num_matches / len(gt_ticks)) if len(gt_ticks) != 0 else 0
    
    return bar_precision, bar_recall, tick_precision, tick_recall

def evaluate_pts_err(gt_bars, gt_ticks, pred_bars, pred_ticks, dist_thresh = 1.5 / 56):
    # fill a matching matrix, check rows (ground truth) for uniqueness
    bar_matches = [[((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) if
                    ((gb[0] - pb[1]) ** 2 + (gb[1] - pb[0]) ** 2) ** (0.5) < dist_thresh
                    else 0
                    for pb in pred_bars] for gb in gt_bars]
    min_bar_dists = [min([bm for bm in row if bm > 0], default = 0.) for row in bar_matches]
    
    tick_matches = [[((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) if
                     ((gt[0] - pt[1]) ** 2 + (gt[1] - pt[0]) ** 2) ** (0.5) < dist_thresh
                     else 0
                     for pt in pred_ticks] for gt in gt_ticks]
    min_tick_dists = [min([tm for tm in row if tm > 0], default = 0.) for row in tick_matches]
    
    return sum(min_bar_dists) + sum(min_tick_dists)

def f1(precision, recall):
    return (2. * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.