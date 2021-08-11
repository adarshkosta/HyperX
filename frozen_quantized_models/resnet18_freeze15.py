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
import pdb
from quant_dorefa import *
__all__ = ['net']


  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):

        residual = x.clone() 
        ################################### 
        out = self.fq15(x)
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu16(out)

        out = self.fq16(out)
        out = self.conv17(out)
        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        residual = out.clone() 

        ################################### 
        #########Layer################ 
        x=out 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn18(x)

        x = self.fq18(x)
        x = self.fc(x)

        x = self.bn19(x)
        x = self.logsoftmax(x)
        return x


class ResNet18(resnet):

    def __init__(self, num_classes=1000, a_bit=7, af_bit=5, w_bit=7, wf_bit=7):
        super(ResNet18, self).__init__()

        self.abit = a_bit
        self.af_bit = af_bit
        self.wbit = w_bit
        self.wf_bit = wf_bit


        QConv2d = conv2d_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)
        QLinear = linear_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)

        self.inflate = 1

        self.fq15 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.conv16=QConv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn16= nn.BatchNorm2d(int(512*self.inflate))
        self.relu16=nn.ReLU(inplace=True)

        self.fq16 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.conv17=QConv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn17= nn.BatchNorm2d(int(512*self.inflate))
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

        self.fq17 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.avgpool=nn.AvgPool2d(7)
        self.bn18= nn.BatchNorm1d(int(512*self.inflate))

        self.fq18 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.fc= QLinear(int(512*self.inflate),num_classes, bias = False)
        self.bn19= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)

def net(**kwargs):
    num_classes, depth, dataset, a_bit, af_bit,w_bit, wf_bit = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'a_bit', 'af_bit', 'w_bit', 'wf_bit'])
    return ResNet18(num_classes=num_classes, a_bit=a_bit, af_bit=af_bit, w_bit=w_bit, wf_bit=wf_bit)

