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
        out = self.conv10(x)
        out = self.bn10(out)
        out = self.relu10(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out+=residual
        out = self.relu11(out)
        residual = out.clone() 
        ################################### 
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu12(out)
        out = self.conv13(out)
        out = self.bn13(out)
        out+=residual
        out = self.relu13(out)
        residual = out.clone() 
        ################################### 
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu14(out)
        out = self.conv15(out)
        out = self.bn15(out)
        residual = self.resconv2(residual)
        out+=residual
        out = self.relu15(out)
        residual = out.clone() 
        ################################### 
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu16(out)
        out = self.conv17(out)
        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        residual = out.clone() 
        ################################### 
        out = self.conv18(out)
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

        self.conv10=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10= nn.BatchNorm2d(32*self.inflate)
        self.relu10=nn.ReLU(inplace=True)
        self.conv11=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11= nn.BatchNorm2d(32*self.inflate)
        self.relu11=nn.ReLU(inplace=True)
        #######################################################

        self.conv12=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(32*self.inflate)
        self.relu12=nn.ReLU(inplace=True)
        self.conv13=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(32*self.inflate)
        self.relu13=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv14=QConv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(64*self.inflate)
        self.relu14=nn.ReLU(inplace=True)
        self.conv15=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(64*self.inflate)
        self.resconv2=nn.Sequential(QConv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(64*self.inflate),)
        self.relu15=nn.ReLU(inplace=True)
        #######################################################

        self.conv16=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(64*self.inflate)
        self.relu16=nn.ReLU(inplace=True)
        self.conv17=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(64*self.inflate)
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

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

        
        
