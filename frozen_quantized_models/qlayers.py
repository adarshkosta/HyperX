import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import *

class QLinear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(QLinear,self).__init__(*args,**kwargs)
        
    def forward(self,input):
        self.weight.data = quantize(self.weight.data, num_bits=8).tensor
        out = F.linear(input,self.weight,None)
        return out
    
class QConv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(QConv2d,self).__init__(*args,**kwargs)

    def forward(self,input):
        self.weight.data = quantize(self.weight.data, num_bits=8).tensor
        out = F.conv2d(input, self.weight, None, self.stride,self.padding, self.dilation, self.groups)
        return out