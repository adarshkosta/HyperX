import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
from qlayers import QConv2d, QLinear
__all__ = ['net']


  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
   
        residual = x.clone() 
        ################################### 
        out = self.conv18(x)
        out = self.bn18(out)
        out = self.relu18(out)
        out = self.conv19(out)
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
        x = self.fc(x)
        x = self.bn21(x)
        x = self.logsoftmax(x)


        return x

class ResNet_cifar(resnet):
    def __init__(self, num_classes=100):
        super(ResNet_cifar, self).__init__()
       
        self.inflate = 1
        self.conv18=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18= nn.BatchNorm2d(64*self.inflate)
        self.relu18=nn.ReLU(inplace=True)
        self.conv19=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19= nn.BatchNorm2d(64*self.inflate)
        self.relu19=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(8)
        self.bn20= nn.BatchNorm1d(64*self.inflate)
        self.fc=QLinear(64*self.inflate,num_classes, bias = False)
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return ResNet_cifar(num_classes=num_classes)

        
        
