# Carlos X. Soto, csoto@bnl.gov, 2022

import os
import time
import matplotlib.pyplot as plt

# read logfile (latest by default) and plot metrics (default to total training and val losses)
def plot_metrics(training_run=None, first=0, plot_accuracy = False):
    # select log lines
    metric_lines = [7, 17]
    metric_names = ['train', 'val']
    metric_scales = [4, 1]          # 4000:1000
#    metric_scales = [192, 54]
    if plot_accuracy:
        metric_lines = [21, 23, 25, 27, 29, 31, 33]
        metric_names = ['bar Precision', 'bar Recall', 'bar F1',
                        'tick Precision', 'tick Recall', 'tick F1', 'mean F1']
        metric_scales = [1, 1, 1, 1, 1, 1, 1]
    log_dir = 'logs'
    if not training_run:
        prevdir = os.getcwd()
        os.chdir(log_dir)
        training_run = sorted(os.listdir('.'), key = os.path.getmtime)[-1]
        os.chdir(prevdir)
    assert os.path.isfile(os.path.join(log_dir, training_run))
    print(os.path.join(log_dir, training_run))
    
    log_lines = open(os.path.join(log_dir, training_run), 'r').read().splitlines()
    metrics = [[float(v) for v in log_lines[mline].split(', ')] for mline in metric_lines]
#    train_losses = [float(v) for v in lines[7].split(', ')]
#    val_losses = [float(v) for v in lines[17].split(', ')]
    
    plt.rcParams["figure.figsize"] = (15,10)
    maxlen = min([len(m) for m in metrics])
    for i, metric in enumerate(metrics):
        plt.plot(range(first, maxlen),[m / metric_scales[i] for m in metric[first:maxlen]], label=metric_names[i])
#    maxlen = min(len(train_losses), len(val_losses))
#    plt.plot(range(first, maxlen),[gps / 4 for gps in train_losses[first:maxlen]], label='train')  #gt
#    plt.plot(range(first, maxlen),[vps * 1 for vps in val_losses[first:maxlen]], label='val')  #val
    plt.legend(loc='upper right');
    plt.show()

plot_metrics()                  # losses
plot_metrics(plot_accuracy = True)   # accuracy metrics (P, R, F1)