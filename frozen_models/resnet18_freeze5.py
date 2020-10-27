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
        out = self.conv6(x)
        out = self.bn6(out)
        residual1 = self.resconv1(residual1)
        out = self.relu6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out+=residual1
        out = self.relu7(out)
        
        
        residual1 = out.clone() 
        ################################### 
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out+=residual1
        out = self.relu9(out)
        
        
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv10(out)
        out = self.bn10(out)
        residual1 = self.resconv2(residual1)
        out = self.relu10(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out+=residual1
        out = self.relu11(out)
        
        
        residual1 = out.clone() 
        ################################### 
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu12(out)
        out = self.conv13(out)
        out = self.bn13(out)
        out+=residual1
        out = self.relu13(out)
        
        
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv14(out)
        out = self.bn14(out)
        residual1 = self.resconv3(residual1)
        out = self.relu14(out)
        out = self.conv15(out)
        out = self.bn15(out)
        out+=residual1
        out = self.relu15(out)
        
        
        residual1 = out.clone() 
        ################################### 
        out = self.conv16(out)
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
        x = self.fc(x)
        x = self.bn19(x)
        x = self.logsoftmax(x)
        return x


class ResNet18(resnet):

    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.inflate = 1

        #######################################################

        self.conv6=nn.Conv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn6= nn.BatchNorm2d(int(128*self.inflate))
        self.resconv1=nn.Sequential(nn.Conv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(128*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu6=nn.ReLU(inplace=True)

        self.conv7=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn7= nn.BatchNorm2d(int(128*self.inflate))
        self.relu7=nn.ReLU(inplace=True)
        #######################################################

        self.conv8=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn8= nn.BatchNorm2d(int(128*self.inflate))
        self.relu8=nn.ReLU(inplace=True)

        self.conv9=nn.Conv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn9= nn.BatchNorm2d(int(128*self.inflate))
        self.relu9=nn.ReLU(inplace=True)
        #######################################################

        self.conv10=nn.Conv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn10= nn.BatchNorm2d(int(256*self.inflate))
        self.resconv2=nn.Sequential(nn.Conv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(256*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu10=nn.ReLU(inplace=True)

        self.conv11=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn11= nn.BatchNorm2d(int(256*self.inflate))
        self.relu11=nn.ReLU(inplace=True)
        #######################################################

        self.conv12=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn12= nn.BatchNorm2d(int(256*self.inflate))
        self.relu12=nn.ReLU(inplace=True)

        self.conv13=nn.Conv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn13= nn.BatchNorm2d(int(256*self.inflate))
        self.relu13=nn.ReLU(inplace=True)
        #######################################################

        self.conv14=nn.Conv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn14= nn.BatchNorm2d(int(512*self.inflate))
        self.resconv3=nn.Sequential(nn.Conv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(512*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu14=nn.ReLU(inplace=True)

        self.conv15=nn.Conv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn15= nn.BatchNorm2d(int(512*self.inflate))
        self.relu15=nn.ReLU(inplace=True)
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
        self.fc= nn.Linear(int(512*self.inflate),num_classes, bias = False)
        self.bn19= nn.BatchNorm1d(num_classes)
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

