#! /usr/bin/env python

import os
from PIL import Image

in_annots_dir = 'augmented_annots'
in_imgs_dir = 'augmented_plots'
out_annots_dir = 'yolo_annots'

if not os.path.isdir(out_annots_dir):
    os.mkdir(out_annots_dir)

#### Non-augmented pyplot parameters (640x480 image) ####
# origin
origin_x = 80
# bars left-most position
bars_xmin = 103
#########################################################

annots = sorted(os.listdir(in_annots_dir))

for a in annots:
    annot_path = os.path.join(in_annots_dir, a)
    img_path = os.path.join(in_imgs_dir, a.replace('txt', 'png'))
    
    # only need image size
    imw = 0
    imh = 0
    with Image.open(img_path) as img:
        imw = img.size[0]
        imh = img.size[1]
    
    point_labels = open(annot_path, 'r').read().splitlines()
    
    orientation = point_labels[0]
    points = point_labels[1:]
    
    origin = ()
    bars = []
    ticks = []
    for p in points:
        c, x, y = p.split()
        if c == 'o':
            origin = (float(x), float(y))
        elif c == 'b':
            bars.append((float(x), float(y)))
        elif c == 't':
            ticks.append((float(x), float(y)))
    # ignore 1st tick (same as origin)
    ticks = ticks[1:]
    
    
    #### map each point to a bounding box
    # for bars: full bar dimension
    # for ticks and origin: fixed width, height of gap between ticks
    
    # offsets from base plots introduced by augmentation script
    augment_offset_x = origin[0] - origin_x
    offset_bars_xmin = bars_xmin + augment_offset_x
    
    # bar width is twice first bar midpoint to left-most edge of first bar
    bar_width = (bars[0][0] - offset_bars_xmin) * 2.0
    # tick vertical spacing is 2nd non-origin tick to origin
    tick_height = origin[1] - ticks[0][1]
    # tick width is small enough to not hit first bar
    tick_width = 40
    
    yolo_annot_path = os.path.join(out_annots_dir, a)
    with open(yolo_annot_path, 'w') as yolo:
    
        # origin is class 0
        o_x = origin[0]
        o_y = origin[1]
        o_w = tick_width
        o_h = tick_height
        #yolo.write(f'0 {o_x} {o_y} {o_w} {o_h}\n')
        yolo.write(f'0 {o_x/imw} {o_y/imh} {o_w/imw} {o_h/imh}\n')
    
        # YOLO bbox annotations are (x-center,y-center, w, h), normalized [0-1]
        for b in bars:
            b_x = b[0]
            b_y = (b[1] + origin[1]) / 2.0
            b_w = bar_width
            b_h = origin[1] - b[1]
            
            # bar is class 1
            #yolo.write(f'1 {b_x} {b_y} {b_w} {b_h}\n')
            yolo.write(f'1 {b_x/imw} {b_y/imh} {b_w/imw} {b_h/imh}\n')
        
        for t in ticks:
            t_x = t[0]
            t_y = t[1]
            t_w = tick_width
            t_h = tick_height
            
            # tick is class 2
            #yolo.write(f'2 {t_x} {t_y} {t_w} {t_h}\n')
            yolo.write(f'2 {t_x/imw} {t_y/imh} {t_w/imw} {t_h/imh}\n')
    
    print(f'converted annotation {a} to YOLO bboxes')