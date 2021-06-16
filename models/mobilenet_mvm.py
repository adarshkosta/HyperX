import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
import numpy as np

import config as cfg

from src.pytorch_mvm_class_v3 import *


__all__ = ['net']

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, tile_size=1):
        self.inflate = 1
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            Conv2d_mvm(in_channels=int(in_planes*self.inflate), out_channels=int(out_planes*self.inflate), 
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=False, 
                    tile_row=tile_size, tile_col=tile_size),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class ConvBNReLU_with_groups(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU_with_groups, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()

    def forward(self, x):
        out = self.ConvBNReLU1(x)
        out = self.InvertedResidual1(out)
        out = self.InvertedResidual2(out)
        residual = out.clone()
        out = self.InvertedResidual3(out) + residual
        out = self.InvertedResidual4(out)
        residual = out.clone()
        out = self.InvertedResidual5(out) + residual
        residual = out.clone()
        out = self.InvertedResidual6(out) + residual
        out = self.InvertedResidual7(out)
        residual = out.clone()
        out = self.InvertedResidual8(out) + residual
        residual = out.clone()
        out = self.InvertedResidual9(out) + residual
        residual = out.clone()
        out = self.InvertedResidual10(out) + residual
        out = self.InvertedResidual11(out)
        residual = out.clone()
        out = self.InvertedResidual12(out) + residual
        residual = out.clone()
        out = self.InvertedResidual13(out) + residual
        out = self.InvertedResidual14(out)
        residual = out.clone()
        out = self.InvertedResidual15(out) + residual
        residual = out.clone()
        out = self.InvertedResidual16(out) + residual
        out = self.InvertedResidual17(out)
        out = self.ConvBNReLU2(out)

        # out = self.avgpool(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        # out = self.dropout1(out)
        out = self.fc(out)

        return out


      
class MobileNet(mobilenet):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()

        self.inflate = 1

        # block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        self.inflate = 1
        self.ConvBNReLU1 = ConvBNReLU(3,32, kernel_size=3, stride=2, tile_size=16)
        self.InvertedResidual1 = nn.Sequential(ConvBNReLU_with_groups(32,32, kernel_size=3, stride=1, groups=32),
                                Conv2d_mvm(int(32*self.inflate), int(16*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=16, tile_col=16),
                                nn.BatchNorm2d(16))
        self.InvertedResidual2 = nn.Sequential(ConvBNReLU(16,96, kernel_size=1, stride=1, tile_size=16),
                                ConvBNReLU_with_groups(96,96, kernel_size=3, stride=2, groups=96),
                                Conv2d_mvm(int(96*self.inflate), int(24*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=8, tile_col=8),
                                nn.BatchNorm2d(24))
        self.InvertedResidual3 = nn.Sequential(ConvBNReLU(24,144, kernel_size=1, stride=1, tile_size=8),
                                ConvBNReLU_with_groups(144,144, kernel_size=3, stride=1, groups=144),
                                Conv2d_mvm(int(144*self.inflate), int(24*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=8, tile_col=8),
                                nn.BatchNorm2d(24))
        self.InvertedResidual4 = nn.Sequential(ConvBNReLU(24,144, kernel_size=1, stride=1, tile_size=8),
                                ConvBNReLU_with_groups(144,144, kernel_size=3, stride=2, groups=144),
                                Conv2d_mvm(int(144*self.inflate), int(32*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=4, tile_col=4),
                                nn.BatchNorm2d(32))
        self.InvertedResidual5 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1, tile_size=4),
                                ConvBNReLU_with_groups(192,192, kernel_size=3, stride=1,groups=192),
                                Conv2d_mvm(int(192*self.inflate), int(32*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=4, tile_col=4),
                                nn.BatchNorm2d(32))
        self.InvertedResidual6 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1, tile_size=4),
                                ConvBNReLU_with_groups(192,192, kernel_size=3, stride=1, groups=192),
                                Conv2d_mvm(int(192*self.inflate), int(32*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=4, tile_col=4),
                                nn.BatchNorm2d(32))
        self.InvertedResidual7 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1, tile_size=4),
                                ConvBNReLU_with_groups(192,192, kernel_size=3, stride=2, groups=192),
                                Conv2d_mvm(int(192*self.inflate), int(64*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(64))
        self.InvertedResidual8 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(384,384, kernel_size=3, stride=1, groups=384),
                                Conv2d_mvm(int(384*self.inflate), int(64*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(64))
        self.InvertedResidual9 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(384,384, kernel_size=3, stride=1, groups=384),
                                Conv2d_mvm(int(384*self.inflate), int(64*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(64))
        self.InvertedResidual10 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(384,384, kernel_size=3, stride=1, groups=384),
                                Conv2d_mvm(int(384*self.inflate), int(64*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(64))
        self.InvertedResidual11 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(384,384, kernel_size=3, stride=1, groups=384),
                                Conv2d_mvm(int(384*self.inflate), int(96*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(96))
        self.InvertedResidual12 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(576,576, kernel_size=3, stride=1, groups=576),
                                Conv2d_mvm(int(576*self.inflate), int(96*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(96))
        self.InvertedResidual13 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(576,576, kernel_size=3, stride=1, groups=576),
                                Conv2d_mvm(int(576*self.inflate), int(96*self.inflate), kernel_size=1, stride=1, bias=False, tile_row=2, tile_col=2),
                                nn.BatchNorm2d(96))
        self.InvertedResidual14 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1, tile_size=2),
                                ConvBNReLU_with_groups(576,576, kernel_size=3, stride=2, groups=576),
                                Conv2d_mvm(int(576*self.inflate), int(160*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=1, tile_col=1),
                                nn.BatchNorm2d(160))
        self.InvertedResidual15 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1, tile_size=1),
                                ConvBNReLU_with_groups(960,960, kernel_size=3, stride=1, groups=960),
                                Conv2d_mvm(int(960*self.inflate), int(160*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=1, tile_col=1),
                                nn.BatchNorm2d(160))
        self.InvertedResidual16 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1, tile_size=1),
                                ConvBNReLU_with_groups(960,960, kernel_size=3, stride=1, groups=960),
                                Conv2d_mvm(int(960*self.inflate), int(160*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=1, tile_col=1),
                                nn.BatchNorm2d(160))
        self.InvertedResidual17 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1, tile_size=1),
                                ConvBNReLU_with_groups(960,960, kernel_size=3, stride=1, groups=960),
                                Conv2d_mvm(int(960*self.inflate), int(320*self.inflate), kernel_size=1, stride=1,  bias=False, tile_row=1, tile_col=1),
                                nn.BatchNorm2d(320))
        

        self.ConvBNReLU2 = ConvBNReLU(320, 1280, kernel_size=1, stride=1, tile_size=1)


        # self.avgpool = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # self.dropout1 = nn.Dropout(0.2)
        self.fc = Linear_mvm(int(1280*self.inflate), num_classes, bias=False)


def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return MobileNet(num_classes=num_classes)

        
