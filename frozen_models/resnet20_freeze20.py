#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:17:22 2020

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
        x = self.bn21(x)
        x = self.logsoftmax(x)

        return x

class ResNet_cifar(resnet):
    def __init__(self, num_classes=100):
        super(ResNet_cifar, self).__init__()
       
        self.inflate = 1
        #########Layer################ 
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return ResNet_cifar(num_classes=num_classes)
        
        
