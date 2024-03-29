import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
import pdb

import config as cfg
from src.pytorch_mvm_class_v3 import *

__all__ = ['net']
  
class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        # print('Input:', torch.mean(x))

        out = self.conv1(x)
        # print('Conv1: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn1(out)
        out = self.relu1(out)
        residual = out.clone() 

        # print('Conv2_In: ', torch.mean(out))
        out = self.conv2(out)
        # print('Conv2_out: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        # print('Conv3: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn3(out)
        out+=residual
        out = self.relu3(out)
        residual = out.clone() 
        ################################### 
        out = self.conv4(out)
        # print('Conv4: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        # print('Conv5: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn5(out)
        out+=residual
        out = self.relu5(out)
        residual = out.clone() 
        ################################### 
        out = self.conv6(out)
        # print('Conv6: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn6(out)
        out = self.relu6(out)
        out = self.conv7(out)
        # print('Conv7: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn7(out)
        out+=residual
        out = self.relu7(out)
        residual = out.clone() 
        ################################### 
        out = self.conv8(out)
        # print('Conv8: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn8(out)
        out = self.relu8(out)
        out = self.conv9(out)
        # print('Conv9: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn9(out)
        residual = self.resconv1(residual)
        out+=residual
        out = self.relu9(out)
        residual = out.clone() 
        ################################### 
        out = self.conv10(out)
        # print('Conv10: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn10(out)
        out = self.relu10(out)
        out = self.conv11(out)
        # print('Conv11: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn11(out)
        out+=residual
        out = self.relu11(out)
        residual = out.clone() 
        ################################### 
        out = self.conv12(out)
        # print('Conv12: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn12(out)
        out = self.relu12(out)
        out = self.conv13(out)
        # print('Conv13: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn13(out)
        out+=residual
        out = self.relu13(out)
        residual = out.clone() 
        ################################### 
        out = self.conv14(out)
        # print('Conv14: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn14(out)
        out = self.relu14(out)
        out = self.conv15(out)
        # print('Conv15: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn15(out)
        residual = self.resconv2(residual)
        out+=residual
        out = self.relu15(out)
        residual = out.clone() 
        ################################### 
        out = self.conv16(out)
        # print('Conv16: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn16(out)
        out = self.relu16(out)
        out = self.conv17(out)
        # print('Conv17: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        residual = out.clone() 
        ################################### 
        out = self.conv18(out)
        # print('Conv18: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn18(out)
        out = self.relu18(out)
        out = self.conv19(out)
        # print('Conv19: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn19(out)
        out+=residual
        out = self.relu19(out)
        residual = out.clone() 
        ################################### 
        
        x=out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn20(x)
        x = self.fc(x)
        # print('FC: ', torch.mean(out))
        # pdb.set_trace()

        x = self.bn21(x)
        x = self.logsoftmax(x)
        
        return x

class ResNet_cifar(resnet):

    def __init__(self, num_classes=100):
        super(ResNet_cifar, self).__init__()
        self.inflate = 1
        
        self.conv1=Conv2d_mvm(3,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(16*self.inflate)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(16*self.inflate)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3= nn.BatchNorm2d(16*self.inflate)
        self.relu3=nn.ReLU(inplace=True)
        #######################################################

        self.conv4=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4= nn.BatchNorm2d(16*self.inflate)
        self.relu4=nn.ReLU(inplace=True)
        self.conv5=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5= nn.BatchNorm2d(16*self.inflate)
        self.relu5=nn.ReLU(inplace=True)
        #######################################################

        self.conv6=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6= nn.BatchNorm2d(16*self.inflate)
        self.relu6=nn.ReLU(inplace=True)
        self.conv7=Conv2d_mvm(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7= nn.BatchNorm2d(16*self.inflate)
        self.relu7=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv8=Conv2d_mvm(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8= nn.BatchNorm2d(32*self.inflate)
        self.relu8=nn.ReLU(inplace=True)
        self.conv9=Conv2d_mvm(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9= nn.BatchNorm2d(32*self.inflate)
        self.resconv1=nn.Sequential(Conv2d_mvm(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(32*self.inflate),)
        self.relu9=nn.ReLU(inplace=True)
        #######################################################

        self.conv10=Conv2d_mvm(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10= nn.BatchNorm2d(32*self.inflate)
        self.relu10=nn.ReLU(inplace=True)
        self.conv11=Conv2d_mvm(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11= nn.BatchNorm2d(32*self.inflate)
        self.relu11=nn.ReLU(inplace=True)
        #######################################################

        self.conv12=Conv2d_mvm(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(32*self.inflate)
        self.relu12=nn.ReLU(inplace=True)
        self.conv13=Conv2d_mvm(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(32*self.inflate)
        self.relu13=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.conv14=Conv2d_mvm(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(64*self.inflate)
        self.relu14=nn.ReLU(inplace=True)
        self.conv15=Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(64*self.inflate)
        self.resconv2=nn.Sequential(Conv2d_mvm(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64*self.inflate),)
        self.relu15=nn.ReLU(inplace=True)
        #######################################################

        self.conv16=Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(64*self.inflate)
        self.relu16=nn.ReLU(inplace=True)
        self.conv17=Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(64*self.inflate)
        self.relu17=nn.ReLU(inplace=True)
        #######################################################

        self.conv18=Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18= nn.BatchNorm2d(64*self.inflate)
        self.relu18=nn.ReLU(inplace=True)
        self.conv19=Conv2d_mvm(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19= nn.BatchNorm2d(64*self.inflate)
        self.relu19=nn.ReLU(inplace=True)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(8)
        self.bn20= nn.BatchNorm1d(64*self.inflate)
        self.fc=Linear_mvm(64*self.inflate,num_classes, bias=False) 
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)

        #self.linear.weight.data = torch.clone(weights_lin)

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
    return ResNet_cifar(num_classes=num_classes)
    #if dataset == 'cifar100':
