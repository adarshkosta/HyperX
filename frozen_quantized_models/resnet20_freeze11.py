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
        out = self.fq11(x)
        out = self.conv12(out)
        # print('Conv12: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn12(out)
        out = self.relu12(out)

        out = self.fq12(out)
        out = self.conv13(out)
        # print('Conv13: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn13(out)
        out+=residual
        out = self.relu13(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq13(out)
        out = self.conv14(out)
        # print('Conv14: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn14(out)
        out = self.relu14(out)

        out = self.fq14(out)
        out = self.conv15(out)
        # print('Conv15: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn15(out)

        r2 = self.fqr2(residual)
        residual = self.resconv2(r2)

        out+=residual
        out = self.relu15(out)
        residual = out.clone() 

        ################################### 
        out = self.fq15(out)
        out = self.conv16(out)
        # print('Conv16: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn16(out)
        out = self.relu16(out)

        out = self.fq16(out)
        out = self.conv17(out)
        # print('Conv17: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        residual = out.clone() 

        ################################### 
        out = self.fq17(out)
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

        self.fq11 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        #######################################################

        self.conv12=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(32*self.inflate)
        self.relu12=nn.ReLU(inplace=True)
        self.fq12 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv13=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(32*self.inflate)
        self.relu13=nn.ReLU(inplace=True)
        self.fq13 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        #########Layer################ 
        self.conv14=QConv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(64*self.inflate)
        self.relu14=nn.ReLU(inplace=True)
        self.fq14 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv15=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(64*self.inflate)
        self.fqr2 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.resconv2=nn.Sequential(QConv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(64*self.inflate),)
        self.relu15=nn.ReLU(inplace=True)
        self.fq15 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv16=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(64*self.inflate)
        self.relu16=nn.ReLU(inplace=True)
        self.fq16 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv17=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(64*self.inflate)
        self.relu17=nn.ReLU(inplace=True)
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
