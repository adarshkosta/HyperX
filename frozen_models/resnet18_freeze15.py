#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 05:56:47 2020

@author: akosta
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
__all__ = ['net']


  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):

        residual1 = x.clone() 
        ################################### 
        out = self.conv16(x)
        out = self.bn16(out)
        out = self.relu16(out)
        out = self.conv17(out)
        out = self.bn17(out)
        out+=residual1
        out = self.relu17(out)
        
        
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        x=out 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn18(x)
        x = self.fc1(x)
        x = self.bnfc1(x)
        x = self.fc2(x)
        x = self.bnfc2(x)
        x = self.logsoftmax(x)
        return x


class ResNet18(resnet):

    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.inflate = 1
        #######################################################

        self.conv16=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn16= nn.BatchNorm2d(int(512*self.inflate))
        self.relu16=nn.ReLU(inplace=True)

        self.conv17=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn17= nn.BatchNorm2d(int(512*self.inflate))
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

        self.avgpool=nn.AvgPool2d(7)
        self.bn18= nn.BatchNorm1d(int(512*self.inflate))
        self.fc1= nn.Linear(512, 256, bias=False)
        self.bnfc1= nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256,num_classes, bias = False)
        self.bnfc2= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


        #init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}

def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return ResNet18(num_classes=num_classes)

