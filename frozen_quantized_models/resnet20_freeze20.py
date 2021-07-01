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

        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset, a_bit, af_bit,w_bit, wf_bit = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'a_bit', 'af_bit', 'w_bit', 'wf_bit'])
    return ResNet_cifar(num_classes=num_classes, a_bit=a_bit, af_bit=af_bit, w_bit=w_bit, wf_bit=wf_bit)
