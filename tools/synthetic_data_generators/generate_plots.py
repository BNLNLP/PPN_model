#! /usr/bin/env python

import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import math

plot_dir = 'plots'
annot_dir = 'annots'


cat_labels = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456-. ()'

num_plots = 5000

num_bars = [2,3,4,5,6,7,8,9,10]

group_ratio = 0.3

num_groups = [2,3,4,5,6,7,8,9,10]
num_bars_per_group = [2,3,4,5]

# plot styles here...

bar_widths = [0.1 * (v + 1) for v in range(10)]

color_options = list(mcolors.CSS4_COLORS.keys())
too_light = ('white', 'whitesmoke', 'snow', 'seashell', 'oldlace', 'floralwhite', 'ivory',
             'honeydew', 'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'lavenderblush',
             'antiquewhite', 'beige', 'bisque', 'mistyrose', 'linen', 'bleachedalmond',
             'papayawhip', 'cornsilk', 'lightyellow', 'lightgoldenrodyellow', 'lightcyan',
             'lavender')
color_options = [c for c in color_options if c not in too_light]

# [vertical] and horizontal
horizontal_ratio = 0.1      # actual sampled is 0.04

# [linear] and log scales...

#val_range_mins = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
val_range_mins = [0]
val_range_maxs = [1, 10, 100, 1000]

for p in range(num_plots):
    print(f'plot {p}')
    # horizontal or vertical
    #vertical = True if random.random() > horizontal_ratio else False
    vertical = True
    
    # grouped or not
    grouped = True if random.random() > group_ratio else False
    
    # number of bars (ungrouped)
    nbars = random.choice(num_bars)
    
    # number of groups, bars/group
    ng = random.choice(num_groups)
    nbg = random.choice(num_bars_per_group)
    
    # data range
    minv = random.choice(val_range_mins)
    maxv = random.choice(val_range_maxs) * random.random()
    diff = maxv - minv
    
    # style
    width = random.choice(bar_widths)
    #width = bar_widths[p % 10]
    #width = 0.01
    colors = []
    if random.random() > 0.5:
        colors = [random.choice(color_options)]
    else:
        colors = [random.choice(color_options) for i in range(nbars)]
    
    labels = [cat_labels[i] for i in range(nbars)]
    values = [(random.random() * diff + minv) for v in range(nbars)]
    
    # generate and save plot
    
    plt.bar(labels, values, width = width, bottom = minv, color=colors)
    plot_id = f'bar_char_{p:06d}'
    #plot_id = f'bar_char_{p:06d}_{width:.1f}_{maxv:.2f}'
    plot_path = os.path.join(plot_dir, plot_id + '.png')
    plt.ylim(minv, maxv)
    yticks = [t for t in list(plt.yticks()[0]) if t >= minv and t <= maxv]
    plt.savefig(plot_path)
    plt.close()
    
##### write annotations (for PyPlot default image size: 640 x 480
    
    # origin
    origin_x = 80
    origin_y = 427
    
    # other end of chart (top-right)
    chart_end_x = 576
    chart_end_y = 58
    
    chart_height = origin_y - chart_end_y
    
    # bar point values
    bars_xmin = 103
    bars_xmax = 552
    bars_xrange = bars_xmax - bars_xmin
    bar_one_xmax = bars_xrange / (nbars * 2)
    # approximately quadratic scaling w.r.t. width between xmin and bar_one_xmax,
    # xoff = -coef*w^2 + (coef + 1)*w, where coef = (1.7/nbars) - 0.17
    master_coef = 1.7
    coef = (master_coef / nbars) - (master_coef / 10)
    bar_one_offset = (-(coef) * width * width + (coef + 1) * width) * bar_one_xmax
    # special-case for 2 bars: sinusoidal error
    if nbars == 2:
        two_bar_error = 2.4 * math.sin(6.28*width - 3.14)
        bar_one_offset -= two_bar_error
    
    first_bar_xpos = bars_xmin + bar_one_offset
    last_bar_xpos = bars_xmax - bar_one_offset
    bar_spaces = (last_bar_xpos - first_bar_xpos) / (nbars - 1)
    
    bars = []
    for i, v in enumerate(values):
        bar_x = first_bar_xpos + bar_spaces * i
        bar_y = origin_y - (v / diff) * chart_height
        bars.append((bar_x, bar_y))
    
    # data tick marks
    dataticks = []
    for t in yticks:
        t_x = origin_x
        t_y = origin_y - (t / diff) * chart_height
        dataticks.append((t_x, t_y))
    
##### write to file (one annotation file per chart)
    
    with open(os.path.join(annot_dir, plot_id + '.txt'), 'w') as annotfile:
        # first line specifies orientation (h/v) of chart
        orientation = 'v' if vertical else 'h'
        annotfile.write(orientation + '\n')
        
        # Annotations are points. Format per line (space-seperated): 
        # class x y
        # class is ['o','t','b'] for 'origin', 'tick', 'bar'
        # x, y are pixel coordinates
        annotfile.write(f'o {origin_x} {origin_y}\n')
        
        for t in dataticks:
            annotfile.write(f't {t[0]} {t[1]}\n')
        for b in bars:
            annotfile.write(f'b {b[0]} {b[1]}\n')
    