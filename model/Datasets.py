# Carlos X. Soto, csoto@bnl.gov, 2022

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding (left, right, top, bottom)
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

# nortmalization for image slices (apply in dataloader)
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# calculated from pyplot generated dataset
normalize = transforms.Normalize(mean=[0.0882, 0.0887, 0.0879], std=[0.0541, 0.0540, 0.0543])
inv_normalize = transforms.Normalize(mean=[-0.0882/0.0541, -0.0887/0.0540, -0.0879/0.0543],
                                     std=[1/0.0541, 1/0.0540, 1/0.0543])


id_to_name = {0: 'None', 1: 'bar', 2: 'tick'}
name_to_id = {'None': 0, 'bar': 1, 'tick': 2}

class BarDataset(Dataset):
    def __init__(self, dataset_list, im_path, annot_path, img_size=224):
        self.ids = open(dataset_list, 'r').read().splitlines()
        self.img_files = [os.path.join(im_path, id) for id in self.ids]
        self.annot_files = [os.path.join(annot_path, id.replace('png', 'txt')) for id in self.ids]
        
        # maybe check that all ids are actually present in im_path and annot_path..
        
        self.img_size = img_size

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        #img.type(torch.cuda.FloatTensor)

        _, h, w = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        
        img = F.interpolate(img.unsqueeze(0), size=224, mode="bilinear", align_corners=False).squeeze(0)
        
        img = normalize(img)        # ASSUMES standard mean and std...

        # ---------
        #  Label
        # ---------
        
        # load annotation file contents
        annot_file = self.annot_files[index % len(self.annot_files)]
        annot_contents = open(annot_file, 'r').read().splitlines()
        orientation = torch.tensor(0.) if annot_contents[0] == 'v' else torch.tensor(1.)
        origin = torch.tensor([float(val) for val in annot_contents[1].split()[1:3]])
        #origin = origin * (224 / 640)
        #origin = origin / torch.tensor([w, h])
        origin = (origin + torch.tensor([pad[0], pad[2]])) / torch.tensor([padded_w, padded_h])
        ticks = []
        bars = []
        for line in annot_contents[2:]:
            point = line.split()
            if point[0] == 't':
                #ticks.append([torch.tensor(float(val)) for val in point[1:3]])
                t = [float(val) for val in point[1:3]]
#                t[0] = t[0] / (w / padded_w) + pad[0]
#                t[1] = t[1] / (h / padded_h) + pad[2]
                ticks.append([torch.tensor(tt) for tt in t])          # TRY REVERSING X/Y
            elif point[0] == 'b':
                #bars.append([torch.tensor(float(val)) for val in point[1:3]])
                b = [float(val) for val in point[1:3]]
#                b[0] = b[0] / (w / padded_w) + pad[0]
#                b[1] = b[1] / (h / padded_h) + pad[2]
                bars.append([torch.tensor(bb) for bb in b])
        
#        print(f'lenB: {len(bars)}, lenT: {len(ticks)}, w: {w}, h: {h}, pad: ', pad)
        # generate GT class and regression feature map        
        gt_cls_map, gt_reg_map = pts_lists_to_map([bars, ticks], w, h, pad)
        
        orientation = orientation.unsqueeze(0)
        
        targets = (orientation, origin, gt_cls_map, gt_reg_map, bars, ticks)
        
        return img_path, img, targets
    
    def __len__(self):
        return len(self.ids)

class ImageOnlyDataset(Dataset):
    def __init__(self, dataset_list, im_path, img_size=224):
        self.ids = open(dataset_list, 'r').read().splitlines()
        self.img_files = [os.path.join(im_path, id) for id in self.ids]
        self.img_size = img_size

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        #img.type(torch.cuda.FloatTensor)

        _, h, w = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        
        img = F.interpolate(img.unsqueeze(0), size=224, mode="bilinear", align_corners=False).squeeze(0)
        img = normalize(img)
        
        return img_path, img
    
    def __len__(self):
        return len(self.ids)

