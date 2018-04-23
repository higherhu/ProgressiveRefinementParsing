#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Higher
@Email: hujiagao@gmail.com

ref: https://github.com/Lextal/pspnet-pytorch
"""
import torch
from torch import nn
from torch.nn import functional as F

from .extractors import *
from .utils import *


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 8 * x.size(2), 8 * x.size(3)
        p = self.conv(x)
        p = F.upsample(input=p, size=(h, w), mode='bilinear')
        return p


class pspnet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = globals()[backend](pretrained)
        self.psp = PSPModule(psp_size, 512, sizes)

        self.final_conv = segnetUp(512, n_classes)


    def forward(self, x):
        f, class_f, _, _ = self.feats(x) # [nb, 512, h/8, w/8]
        p = self.psp(f)

        p = self.final_conv(p)
        p = F.upsample(input=p, size=x.data.shape[2:4], mode='bilinear')

        return p

class pspnet_stack(nn.Module):
    def __init__(self, n_classes=(3, 6, 11), sizes=(1, 2, 3, 6), psp_size=512, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = globals()[backend](pretrained)
        self.psp = PSPModule(psp_size, 512, sizes)

        self.final_conv = segnetUp(512, n_classes[0])

        self.final_1 = segnetUp(512+n_classes[0]+256, n_classes[1])
        self.final_2 = segnetUp(512+n_classes[1]+64, n_classes[2])

    def forward(self, x):
        f, f_3, f_2, f_1 = self.feats(x) # f:[nb, 512, h/8, w/8]; f_3:[nb, 256, h/8, w/8]; f_2:[nb, 128, h/8, w/8], f_1:[nb, 64, h/4, w/4]

        p_0_pre = self.psp(f)

        p_0 = self.final_conv(p_0_pre)

        p_0_cat = torch.cat([(f),
                             p_0,
                             (f_3)],
                            dim=1)
        p_1_pre = p_0_cat # self.psp_1(p_0_cat)
        p_1 = self.final_1(p_1_pre)

        p_1_cat = torch.cat([nn.Upsample(size=f_1.data.shape[2:4], mode='bilinear')(f),
                             nn.Upsample(size=f_1.data.shape[2:4], mode='bilinear')(p_1),
                             (f_1)],
                            dim=1)
        p_2_pre = p_1_cat #self.psp_1(p_1_cat)
        p_2 = self.final_2(p_2_pre)

        p_0 = F.upsample(input=p_0, size=x.data.shape[2:4], mode='bilinear')
        p_1 = F.upsample(input=p_1, size=x.data.shape[2:4], mode='bilinear')
        p_2 = F.upsample(input=p_2, size=x.data.shape[2:4], mode='bilinear')

        return p_0, p_1, p_2

def PSPNet(nIn=3, num_classes=21, pretrain=False):
    model = pspnet(n_classes=num_classes, backend='resnet18', pretrained=pretrain)

    return model

def PSPNet_Stack(nIn=3, num_classes=(3, 6, 11), pretrain=False):
    model = pspnet_stack(n_classes=num_classes, backend='resnet18', pretrained=pretrain)

    return model