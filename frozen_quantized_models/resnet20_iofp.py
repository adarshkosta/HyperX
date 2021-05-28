import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
from quant_dorefa import *
__all__ = ['net']

class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        x = self.fq0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fq1(x)

        residual = x.clone() 
        out = x.clone()

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fq2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out+=residual
        out = self.relu3(out)
        out = self.fq3(out)

        residual = out.clone() 
        ################################### 
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.fq4(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out+=residual
        out = self.relu5(out)
        out = self.fq5(out)

        residual = out.clone() 
        ################################### 
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.fq6(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out+=residual
        out = self.relu7(out)
        out = self.fq7(out)

        residual = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)
        out = self.fq8(out)

        out = self.conv9(out)
        out = self.bn9(out)
        residual = self.resconv1(residual)
        out+=residual
        out = self.relu9(out)
        out = self.fq9(out)

        residual = out.clone() 
        ################################### 
        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu10(out)
        out = self.fq10(out)

        out = self.conv11(out)
        out = self.bn11(out)
        out+=residual
        out = self.relu11(out)
        out = self.fq11(out)

        residual = out.clone() 
        ################################### 
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu12(out)
        out = self.fq12(out)

        out = self.conv13(out)
        out = self.bn13(out)
        out+=residual
        out = self.relu13(out)
        out = self.fq13(out)

        residual = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.conv14(out)
        out = self.bn14(out)
        out = self.relu14(out)
        out = self.fq14(out)

        out = self.conv15(out)
        out = self.bn15(out)
        residual = self.resconv2(residual)
        out+=residual
        out = self.relu15(out)
        out = self.fq15(out)

        residual = out.clone() 
        ################################### 
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu16(out)
        out = self.fq16(out)

        out = self.conv17(out)
        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        out = self.fq17(out)

        residual = out.clone() 
        ################################### 
        out = self.conv18(out)
        out = self.bn18(out)
        out = self.relu18(out)
        out = self.fq18(out)

        out = self.conv19(out)
        out = self.bn19(out)
        out+=residual
        out = self.relu19(out)
        out = self.fq19(out)

        residual = out.clone() 
        ################################### 
        #########Layer################ 
        x=out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn20(x)
        x = self.fq20(x)

        x = self.fc(x)
        x = self.bn21(x)
        x = self.logsoftmax(x)
        return x

class ResNet_cifar(resnet):

    def __init__(self, num_classes=100):
        super(ResNet_cifar, self).__init__()

        self.wbit = 8
        self.abit = 8

        QConv2d = conv2d_Q_fn(w_bit=self.wbit)
        QLinear = linear_Q_fn(w_bit=self.wbit)

        QConv2d_fp = conv2d_Q_fn(w_bit=16)
        QLinear_fp = linear_Q_fn(w_bit=16)
        

        self.inflate = 1
        self.fq0 = activation_quantize_fn(a_bit=self.abit)
        self.conv1=nn.Conv2d(3,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(16*self.inflate)
        self.relu1=nn.ReLU(inplace=True)
        self.fq1 = activation_quantize_fn(a_bit=self.abit)

        self.conv2=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(16*self.inflate)
        self.relu2=nn.ReLU(inplace=True)
        self.fq2 = activation_quantize_fn(a_bit=self.abit)

        self.conv3=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3= nn.BatchNorm2d(16*self.inflate)
        self.relu3=nn.ReLU(inplace=True)
        self.fq3 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv4=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4= nn.BatchNorm2d(16*self.inflate)
        self.relu4=nn.ReLU(inplace=True)
        self.fq4 = activation_quantize_fn(a_bit=self.abit)

        self.conv5=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5= nn.BatchNorm2d(16*self.inflate)
        self.relu5=nn.ReLU(inplace=True)
        self.fq5 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv6=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6= nn.BatchNorm2d(16*self.inflate)
        self.relu6=nn.ReLU(inplace=True)
        self.fq6 = activation_quantize_fn(a_bit=self.abit)

        self.conv7=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7= nn.BatchNorm2d(16*self.inflate)
        self.relu7=nn.ReLU(inplace=True)
        self.fq7 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        #########Layer################ 
        self.conv8=QConv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8= nn.BatchNorm2d(32*self.inflate)
        self.relu8=nn.ReLU(inplace=True)
        self.fq8 = activation_quantize_fn(a_bit=self.abit)

        self.conv9=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9= nn.BatchNorm2d(32*self.inflate)
        self.resconv1=nn.Sequential(QConv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(32*self.inflate),)
        self.relu9=nn.ReLU(inplace=True)
        self.fq9 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv10=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10= nn.BatchNorm2d(32*self.inflate)
        self.relu10=nn.ReLU(inplace=True)
        self.fq10 = activation_quantize_fn(a_bit=self.abit)

        self.conv11=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11= nn.BatchNorm2d(32*self.inflate)
        self.relu11=nn.ReLU(inplace=True)
        self.fq11 = activation_quantize_fn(a_bit=self.abit)

        #######################################################

        self.conv12=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(32*self.inflate)
        self.relu12=nn.ReLU(inplace=True)
        self.fq12 = activation_quantize_fn(a_bit=self.abit)

        self.conv13=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(32*self.inflate)
        self.relu13=nn.ReLU(inplace=True)
        self.fq13 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        #########Layer################ 
        self.conv14=QConv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(64*self.inflate)
        self.relu14=nn.ReLU(inplace=True)
        self.fq14 = activation_quantize_fn(a_bit=self.abit)

        self.conv15=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(64*self.inflate)
        self.resconv2=nn.Sequential(QConv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(64*self.inflate),)
        self.relu15=nn.ReLU(inplace=True)
        self.fq15 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv16=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(64*self.inflate)
        self.relu16=nn.ReLU(inplace=True)
        self.fq16 = activation_quantize_fn(a_bit=self.abit)

        self.conv17=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(64*self.inflate)
        self.relu17=nn.ReLU(inplace=True)
        self.fq17 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv18=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18= nn.BatchNorm2d(64*self.inflate)
        self.relu18=nn.ReLU(inplace=True)
        self.fq18 = activation_quantize_fn(a_bit=self.abit)

        self.conv19=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19= nn.BatchNorm2d(64*self.inflate)
        self.relu19=nn.ReLU(inplace=True)
        self.fq19 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(8)
        self.bn20= nn.BatchNorm1d(64*self.inflate)
        self.fq20 = activation_quantize_fn(a_bit=self.abit)

        self.fc=nn.Linear(64*self.inflate,num_classes, bias=False)
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return ResNet_cifar(num_classes=num_classes)
