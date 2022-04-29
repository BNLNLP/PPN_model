# Carlos X. Soto, csoto@bnl.gov, 2022

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BarValueExtractor(nn.Module):
    def __init__(self):
        super(BarValueExtractor, self).__init__()        
        
        resnet50 = models.resnet50(pretrained=True)
        # resnet backbone 1 (256 channels, downsampled x4 -> 56x56)
        self.features1 = nn.Sequential(*list(resnet50.children())[:-5])
        # resnet backbone 1 (2048 channels, downsampled x32 -> 7x7)
        self.features2 = nn.Sequential(*list(resnet50.children())[-5:-2])
        
        # to combine deep and shallow feature maps
        self.downchannel = nn.Conv2d(2048, 256, 1, 1, 0)
        self.upsample = nn.Upsample(size=(56,56), mode='bilinear', align_corners=True)
        #self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')
        
        # freeze paramters at start of network
        for param in self.features1.parameters():
            param.requires_grad = False
#        for param in self.features2.parameters():
#            param.requires_grad = False
            
        # various heads; expect 56x56 feature map (256-channel)
        
        self.to_single_channel = nn.Conv2d(256, 1, 1, 1, 0)
        self.origin = nn.Linear(3136, 2)
#        self.origin = nn.Sequential(                         # 56x56 to 14x14?
#                            nn.Linear(3136, 196),
#                            nn.Linear(196,2))
#        self.orientation = nn.Linear(3136, 1)                # don't need right now...
        
        # wider receptive field (9x9 kernel), pad to keep same size featmap
        #self.ppn_feats = nn.Conv2d(256, 256, 9, 1, 4)
        self.ppn_feats = nn.Sequential(
                            nn.Conv2d(256,256,3,padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(256,256,3,padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(256,256,3,padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(256,256,3,padding=1),
                            nn.LeakyReLU())
        
        # output 56x56 pixel map of point classes ['None','t','b'], and regression values (dx, dy)
        self.pnt_class = nn.Conv2d(256, 3, 1, 1, 0)        
        self.pnt_reg = nn.Conv2d(256, 2, 1, 1, 0)
        
        self.drop_layer = nn.Dropout(p=0.5)
        
    def forward(self, x):
        
        #print('x.shape', x.shape)
        
        f1 = self.features1(x)
        f2 = self.features2(f1)
        feat = f1 + self.upsample(self.downchannel(f2))  # 256-channel, 56x56
        #return feat
        
        # this needs a relu activation here, e.g.:
        feat = F.leaky_relu(feat)
        
        feat = self.drop_layer(feat)
        
        flat56 = torch.flatten(self.to_single_channel(feat), start_dim=1)        
        #return flat56
        
        origin = self.origin(flat56)
        origin_pred = torch.sigmoid(origin)           # sigmoid might not be needed?
#        orient = self.orientation(flat56)
#        orient_pred = torch.sigmoid(orient)           # sigmoid might not be needed?
        
        orient_pred = origin_pred
        #return origin_pred, orient_pred
        
        pnt_feats = self.ppn_feats(feat)        
        #return pnt_feats
        
        pnt_feats = self.drop_layer(pnt_feats)

        # TODO: LOSS function needs to incorporate some threshold for class prediciont,
        # and ignore regression loss for points that don't meed threshold
        pts_cls_pred = self.pnt_class(pnt_feats)
        #pts_cls_prob = F.softmax(pts_cls, dim=1)     # DON'T NORMALIZE FOR PYTORCH CROSS-ENTROPY LOSS
        pts_reg = self.pnt_reg(pnt_feats)
        pts_reg_pred = torch.tanh(pts_reg)
        
        # No NMS right now: 1 point per pixel only
        
        # TODO: conversion to points list may need adjustment for non-softmax-ed classes (e.g. some threshold?)
        # extact bar, tick predictions
        bars = []
        ticks = []
#        bars, ticks = pts_map_to_lists(pts_cls_pred, pts_reg_pred)
        
        return orient_pred, origin_pred, bars, ticks, pts_cls_pred, pts_reg_pred