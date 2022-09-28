# Carlos X. Soto, csoto@bnl.gov, 2022

import torch
from torch.utils.data import DataLoader
import datetime

from model.Datasets import BarDataset, ImageOnlyDataset

#checkpoint_dir = 'checkpoints'

valid_datasets = ('augmented_bars', 'generated_bars',
                  'generated_pies', 'manually_annotated',
                  'zhou_2021')

def get_dataloaders(dataset_id, bs):
    assert dataset_id in valid_datasets, 'invalid dataset ID'
    
    im_path = f'datasets/{dataset_id}/imgs'
    annot_path = f'datasets/{dataset_id}/annots'
    train_list = f'datasets/{dataset_id}/train.txt'
    val_list = f'datasets/{dataset_id}/val.txt'
    
    train_set = BarDataset(train_list, im_path, annot_path)
    val_set = BarDataset(val_list, im_path, annot_path)
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = bs, num_workers=8, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = bs, num_workers=8, shuffle = False)
    
    return train_dataloader, val_dataloader

'''
# Dataset and dataloader
train_list = 'datasets/generated_bars/train.txt'
test_list = 'datasets/generated_bars/test.txt'
im_path = 'datasets/generated_bars/imgs'
annot_path = 'datasets/generated_bars/annots'

# new, isolated testset (not to be used for HPO)
test2_im_path = 'generated_plots/testset_plots'
test2_annot_path = 'generated_plots/testset_annots'
test2_list = 'generated_plots/test2.txt'

# un-annotated bio images
bio_impath = 'yolo/data/biocharts/images'
bio_imlist = 'yolo/data/biocharts/imagelist_nameonly.txt'

# annotated bio images (N=95)
bioannot_impath = 'annotated_plots/v2_246charts/imgs246'
bioannot_annotpath = 'annotated_plots/v2_246charts/m246_annots'
bioannot_train = 'annotated_plots/v2_246charts/m246_train.txt'
bioannot_test = 'annotated_plots/v2_246charts/m246_test.txt'

# real-augmented synthetic plots
real_aug_impath = 'real-augmented_plots/plots'
real_aug_annotpath = 'real-augmented_plots/annots'
real_aug_train = 'real-augmented_plots/train.txt'
real_aug_val = 'real-augmented_plots/val.txt'

# Zhao bar chart dataset
z_impath = 'zhao_dataset/imgs'
z_annotpath = 'zhao_dataset/annots'
z_trainfull = 'zhao_dataset/train.txt'
z_train = 'zhao_dataset/train4k.txt'
z_test = 'zhao_dataset/test.txt'


# datasets
train_set = BarDataset(dataset_list, im_path, annot_path)
test_set = BarDataset(test_list, im_path, annot_path)
bio_set = ImageOnlyDataset(bio_imlist, bio_impath)
test2_set = BarDataset(test2_list, test2_im_path, test2_annot_path)

train4_set = BarDataset(bioannot_train, bioannot_impath, bioannot_annotpath)
test4_set = BarDataset(bioannot_test, bioannot_impath, bioannot_annotpath)

train_aug_set = BarDataset(real_aug_train, real_aug_impath, real_aug_annotpath)
val_aug_set = BarDataset(real_aug_val, real_aug_impath, real_aug_annotpath)

train_z_set = BarDataset(z_train, z_impath, z_annotpath)
test_z_set = BarDataset(z_test, z_impath, z_annotpath)


# batch size
bs = 1

# dataloaders
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = bs, num_workers=8, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = bs, num_workers=8, shuffle = True)
bioimg_dataloader = torch.utils.data.DataLoader(bio_set, batch_size = bs, num_workers=8, shuffle = True)
test2_dataloader = torch.utils.data.DataLoader(test2_set, batch_size = bs, num_workers=8, shuffle = False)

train4_dataloader = torch.utils.data.DataLoader(train4_set, batch_size = bs, num_workers=8, shuffle = False)
test4_dataloader = torch.utils.data.DataLoader(test4_set, batch_size = bs, num_workers=8, shuffle = False)

train_aug_dataloader = torch.utils.data.DataLoader(train_aug_set, batch_size = bs, num_workers=8, shuffle = False)
val_aug_dataloader = torch.utils.data.DataLoader(val_aug_set, batch_size = bs, num_workers=8, shuffle = False)

train_z_dataloader = torch.utils.data.DataLoader(train_z_set, batch_size = bs, num_workers=8, shuffle = True)
test_z_dataloader = torch.utils.data.DataLoader(test_z_set, batch_size = bs, num_workers=8, shuffle = False)
'''


# LOG LOSSES AND BATCH TRAINING TIMES

def log_losses(start_epoch, num_epochs,
               origin_losses, pclass_losses, pntreg_losses, losses, train_times,
               vorigin_losses, vpclass_losses, vpntreg_losses, vlosses, vtimes,
               barPs, barRs, barF1s, tickPs, tickRs, tickF1s, meanF1s):
    timestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfn = f'logs/{timestring}_train_epochs_{start_epoch}-{num_epochs}.log'
    with open(logfn,'w') as logfile:
        logfile.write('Training origin losses:\n')
        logfile.write(', '.join([str(item) for item in origin_losses]))
    #    logfile.write('\nTraining orient losses:\n')
    #    logfile.write(', '.join([str(item) for item in orient_losses]))
        logfile.write('\nTraining point classification losses:\n')
        logfile.write(', '.join([str(item) for item in pclass_losses]))
        logfile.write('\nTraining point regression losses:\n')
        logfile.write(', '.join([str(item) for item in pntreg_losses]))
        logfile.write('\nTraining total losses:\n')
        logfile.write(', '.join([str(item) for item in losses]))
        logfile.write('\nTraining times:\n')
        logfile.write(', '.join([str(item) for item in train_times]))
        
        logfile.write('\nTest origin losses:\n')
        logfile.write(', '.join([str(item) for item in vorigin_losses]))
        logfile.write('\nTest orient losses:\n')
    #    logfile.write(', '.join([str(item) for item in vorient_losses]))
    #    logfile.write('\nTest point classification losses:\n')
        logfile.write(', '.join([str(item) for item in vpclass_losses]))
        logfile.write('\nTest point regression losses:\n')
        logfile.write(', '.join([str(item) for item in vpntreg_losses]))
        logfile.write('\nTest total losses:\n')
        logfile.write(', '.join([str(item) for item in vlosses]))
        logfile.write('\nTest times:\n')
        logfile.write(', '.join([str(item) for item in vtimes]))
        
        logfile.write('\nBar Precisions:\n')
        logfile.write(', '.join([str(item) for item in barPs]))
        logfile.write('\nBar Recalls:\n')
        logfile.write(', '.join([str(item) for item in barRs]))
        logfile.write('\nBar F1s:\n')
        logfile.write(', '.join([str(item) for item in barF1s]))
        logfile.write('\nTick Precisions:\n')
        logfile.write(', '.join([str(item) for item in tickPs]))
        logfile.write('\nTick Recalls:\n')
        logfile.write(', '.join([str(item) for item in tickRs]))
        logfile.write('\nTick F1s:\n')
        logfile.write(', '.join([str(item) for item in tickF1s]))
        
        logfile.write('\nMean F1s:\n')
        logfile.write(', '.join([str(item) for item in meanF1s]))