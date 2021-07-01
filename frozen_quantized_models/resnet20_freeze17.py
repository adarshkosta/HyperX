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
        out = self.fq17(x)
        out = self.conv18(out)
        # print('Conv18: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn18(out)
        out = self.relu18(out)

        out = self.fq18(out)
        out = self.conv19(out)
        # print('Conv19: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn19(out)
        out+=residual
        out = self.relu19(out)
        residual = out.clone() 
        
        ################################### 
        #########Layer################ 
        x=out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn20(x)

        x = self.fq19(x)
        x = self.fc(x)
        
        # print('FC: ', torch.mean(out))
        # pdb.set_trace()

        x = self.bn21(x)
        x = self.logsoftmax(x)
        return x

class ResNet_cifar(resnet):

    def __init__(self, num_classes=100, a_bit=7, af_bit=4, w_bit=7, wf_bit=6):
        super(ResNet_cifar, self).__init__()

        
        self.abit = a_bit
        self.af_bit = af_bit
        self.wbit = w_bit
        self.wf_bit = wf_bit


        QConv2d = conv2d_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)
        QLinear = linear_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)
        

        self.inflate = 1

        self.fq17 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv18=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18= nn.BatchNorm2d(64*self.inflate)
        self.relu18=nn.ReLU(inplace=True)
        self.fq18 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv19=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19= nn.BatchNorm2d(64*self.inflate)
        self.relu19=nn.ReLU(inplace=True)
        self.fq19 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(8)
        self.bn20= nn.BatchNorm1d(64*self.inflate)
        self.fq20 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.fc=QLinear(64*self.inflate,num_classes, bias=False)
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset, a_bit, af_bit,w_bit, wf_bit = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'a_bit', 'af_bit', 'w_bit', 'wf_bit'])
    return ResNet_cifar(num_classes=num_classes, a_bit=a_bit, af_bit=af_bit, w_bit=w_bit, wf_bit=wf_bit)
