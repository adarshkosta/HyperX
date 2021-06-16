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
        out = self.maxpool(x)

        residual1 = out.clone() 
        
        out = self.fq1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fq2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out+=residual1
        out = self.relu3(out)
        
        residual1 = out.clone() 
        ################################### 
        out = self.fq3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fq4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out+=residual1
        out = self.relu5(out)
        
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.fq5(out)
        out = self.conv6(out)
        out = self.bn6(out)

        r1 = self.fqr1(residual1)
        residual1 = self.resconv1(r1)

        out = self.relu6(out)

        out = self.fq6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out+=residual1
        out = self.relu7(out)
        
        residual1 = out.clone() 
        ################################### 
        out = self.fq7(out)
        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu8(out)

        out = self.fq8(out)
        out = self.conv9(out)
        out = self.bn9(out)
        out+=residual1
        out = self.relu9(out)

        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.fq9(out)
        out = self.conv10(out)
        out = self.bn10(out)

        r2 = self.fqr2(residual1)
        residual1 = self.resconv2(r2)
        out = self.relu10(out)

        out = self.fq10(out)
        out = self.conv11(out)
        out = self.bn11(out)
        out+=residual1
        out = self.relu11(out)
        
        residual1 = out.clone() 
        ################################### 
        out = self.fq11(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu12(out)

        out = self.fq12(out)
        out = self.conv13(out)
        out = self.bn13(out)
        out+=residual1
        out = self.relu13(out)
        
        residual1 = out.clone() 
        ################################### 
        #########Layer################ 
        out = self.fq13(out)
        out = self.conv14(out)
        out = self.bn14(out)

        r3 = self.fqr3(residual1)
        residual1 = self.resconv3(r3)
        out = self.relu14(out)

        out = self.fq14(out)
        out = self.conv15(out)
        out = self.bn15(out)
        out+=residual1
        out = self.relu15(out)
          
        residual1 = out.clone() 
        ################################### 
        out = self.fq15(out)
        out = self.conv16(out)
        out = self.bn16(out)
        out = self.relu16(out)

        out = self.fq16(out)
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
        
        x = self.fq17(x)
        x = self.fc(x)

        x = self.bn19(x)
        x = self.logsoftmax(x)
        return x


class ResNet18(resnet):

    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        self.wbit = 7
        self.abit = 7

        QConv2d = conv2d_Q_fn(w_bit=self.wbit)
        QLinear = linear_Q_fn(w_bit=self.wbit)

        self.inflate = 1
        self.fq0 = activation_quantize_fn(a_bit=self.abit)
        self.conv1=QConv2d(3,int(64*self.inflate), kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1= nn.BatchNorm2d(int(64*self.inflate))
        self.relu1=nn.ReLU(inplace=True)
        self.fq1 = activation_quantize_fn(a_bit=self.abit)
        
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2=QConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn2= nn.BatchNorm2d(int(64*self.inflate))
        self.relu2=nn.ReLU(inplace=True)
        self.fq2 = activation_quantize_fn(a_bit=self.abit)

        self.conv3=QConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn3= nn.BatchNorm2d(int(64*self.inflate))
        self.relu3=nn.ReLU(inplace=True)
        self.fq3 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv4=QConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn4= nn.BatchNorm2d(int(64*self.inflate))
        self.relu4=nn.ReLU(inplace=True)
        self.fq4 = activation_quantize_fn(a_bit=self.abit)

        self.conv5=QConv2d(int(64*self.inflate), int(64*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn5= nn.BatchNorm2d(int(64*self.inflate))
        self.relu5=nn.ReLU(inplace=True)
        self.fq5 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv6=QConv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn6= nn.BatchNorm2d(int(128*self.inflate))

        self.fqr1 = activation_quantize_fn(a_bit=self.abit)
        self.resconv1=nn.Sequential(QConv2d(int(64*self.inflate), int(128*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(128*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu6=nn.ReLU(inplace=True)
        self.fq6 = activation_quantize_fn(a_bit=self.abit)

        self.conv7=QConv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn7= nn.BatchNorm2d(int(128*self.inflate))
        self.relu7=nn.ReLU(inplace=True)
        self.fq7 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv8=QConv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn8= nn.BatchNorm2d(int(128*self.inflate))
        self.relu8=nn.ReLU(inplace=True)
        self.fq8 = activation_quantize_fn(a_bit=self.abit)

        self.conv9=QConv2d(int(128*self.inflate), int(128*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn9= nn.BatchNorm2d(int(128*self.inflate))
        self.relu9=nn.ReLU(inplace=True)
        self.fq9 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv10=QConv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn10= nn.BatchNorm2d(int(256*self.inflate))

        self.fqr2 = activation_quantize_fn(a_bit=self.abit)
        self.resconv2=nn.Sequential(QConv2d(int(128*self.inflate), int(256*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(256*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu10=nn.ReLU(inplace=True)
        self.fq10 = activation_quantize_fn(a_bit=self.abit)

        self.conv11=QConv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn11= nn.BatchNorm2d(int(256*self.inflate))
        self.relu11=nn.ReLU(inplace=True)
        self.fq11 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv12=QConv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn12= nn.BatchNorm2d(int(256*self.inflate))
        self.relu12=nn.ReLU(inplace=True)
        self.fq12 = activation_quantize_fn(a_bit=self.abit)

        self.conv13=QConv2d(int(256*self.inflate), int(256*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn13= nn.BatchNorm2d(int(256*self.inflate))
        self.relu13=nn.ReLU(inplace=True)
        self.fq13 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv14=QConv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=3, stride=2, padding=1, bias = False)
        self.bn14= nn.BatchNorm2d(int(512*self.inflate))

        self.fqr3 = activation_quantize_fn(a_bit=self.abit)
        self.resconv3=nn.Sequential(QConv2d(int(256*self.inflate), int(512*self.inflate), kernel_size=1, stride=2, padding=0, bias = False),
        nn.BatchNorm2d(int(512*self.inflate)),
        nn.ReLU(inplace=True),)
        self.relu14=nn.ReLU(inplace=True)
        self.fq14 = activation_quantize_fn(a_bit=self.abit)

        self.conv15=QConv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn15= nn.BatchNorm2d(int(512*self.inflate))
        self.relu15=nn.ReLU(inplace=True)
        self.fq15 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.conv16=QConv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn16= nn.BatchNorm2d(int(512*self.inflate))
        self.relu16=nn.ReLU(inplace=True)
        self.fq16 = activation_quantize_fn(a_bit=self.abit)

        self.conv17=QConv2d(int(512*self.inflate), int(512*self.inflate), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn17= nn.BatchNorm2d(int(512*self.inflate))
        self.relu17=nn.ReLU(inplace=True)
        self.fq17 = activation_quantize_fn(a_bit=self.abit)
        #######################################################

        self.avgpool=nn.AvgPool2d(7)
        self.bn18= nn.BatchNorm1d(int(512*self.inflate))
        self.fq18 = activation_quantize_fn(a_bit=self.abit)

        self.fc= QLinear(int(512*self.inflate),num_classes, bias = False)
        self.bn19= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)

def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return ResNet18(num_classes=num_classes)

