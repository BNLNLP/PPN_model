#! /usr/bin/env python

# only simple overcropping augmentations

import os
from PIL import Image
import random

src_annot_dir = 'annots'
src_plots_dir = 'plots'

dest_annot_dir = 'augmented_annots'
dest_plots_dit = 'augmented_plots'

if not os.path.isdir(dest_annot_dir):
    os.mkdir(dest_annot_dir)
if not os.path.isdir(dest_plots_dit):
    os.mkdir(dest_plots_dit)

annots = sorted(os.listdir(src_annot_dir))

generated = True

# for generated plots (640x480) scaled 1.5x (960x720)
# (x0,y0,w,h)
minsafe_crop = (26+160, 46+120, 553, 408)

min_x0 = 0
max_x0 = minsafe_crop[0]
min_x1 = minsafe_crop[0] + minsafe_crop[2]
max_x1 = 960
min_y0 = 0
max_y0 = minsafe_crop[1]
min_y1 = minsafe_crop[1] + minsafe_crop[3]
max_y1 = 720

print('x0 range:', min_x0, max_x0)
print('y0 range:', min_y0, max_y0)
print('x1 range:', min_x1, max_x1)
print('y1 range:', min_y1, max_y1)

max_w = int(minsafe_crop[2] * 1.5)
max_h = int(minsafe_crop[3] * 1.5)


# generated plots are 640 x 480
# min safe crop on generated plots (x0,y0,w,h): [26, 46, 553, 408]
# max crop (x1.5) (x0,y0,w,h): [-160, -120, 830, 612]

for a in annots:
    img_path = os.path.join(src_plots_dir, a.replace('txt', 'png'))
    img = Image.open(img_path)
    
    # pad image 1.5x to keep white background
    paddedimg = Image.new(img.mode, (960, 720), (255, 255, 255))
    paddedimg.paste(img, (160, 120))
    
    # generate random crops in specified ranges
    x0 = random.randrange(min_x0, max_x0)
    y0 = random.randrange(min_y0, max_y0)
    x1 = random.randrange(min_x1, max_x1)
    y1 = random.randrange(min_y1, max_y1)
    
    #print(x0, y0, x1, y1)
    
    crop = paddedimg.crop((x0, y0, x1, y1))
    
    crop_path = os.path.join(dest_plots_dit, a.replace('txt', 'png'))
    crop.save(crop_path)
    
    # re-compute new (crop+shift augmented) annotation coords
    labels = open(os.path.join(src_annot_dir, a), 'r').read().splitlines()
    
    aug_annot_path = os.path.join(dest_annot_dir, a)
    with open(aug_annot_path, 'w') as out_annot:
        out_annot.write(labels[0] + '\n')
        
        for lab in labels[1:]:
            c, x, y = lab.split()
        
            aug_x = float(x) + 160.0 - x0
            aug_y = float(y) + 120.0 - y0
            aug_label = f'{c} {aug_x:.2f} {aug_y:.2f}\n'
            
            out_annot.write(aug_label)
    
    print(f'augmented {a}')