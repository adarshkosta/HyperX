import torch.nn as nn
import torch
import torch.nn.functional as F
import os

if_bit_slicing = True
debug = True

mvm = False
non_ideality = False

inmax_test = 1.4 #1.2 #1.4
inmin_test = 0.826 #0.857 #0.826

## Use global parameters (below) for all layers or layer specific parameters
val = True
ifglobal_weight_bits = val
ifglobal_weight_bit_frac = val
ifglobal_input_bits = val
ifglobal_input_bit_frac = val
ifglobal_xbar_col_size = val
ifglobal_xbar_row_size = val
ifglobal_tile_col = False
ifglobal_tile_row = False
ifglobal_bit_stream = val
ifglobal_bit_slice = val
ifglobal_adc_bit = val
ifglobal_acm_bits = val
ifglobal_acm_bit_frac = val
ifglobal_xbmodel = val
ifglobal_xbmodel_weight_path = val
ifglobal_dataset = True  # if True data collected from all layers

## Fixed point arithmetic configurations
weight_bits = 8 #16 #8 
weight_bit_frac = 7 #12 #6
input_bits = 8
input_bit_frac = 5

## Tiling configurations
tile_row = 2
tile_col = 2
xbar_row_size = 128 #128
xbar_col_size = 128 #128 #keep same as row, heterogenous not supported

## Bit-slicing configurations
bit_stream = 1
bit_slice = 2
adc_bit = 14
acm_bits = 32
acm_bit_frac = 24

## GENIEx configurations
loop = False # executes GENIEx with batching when set to False

## GENIEx data collection configuations
dataset = False
direc = 'geniex_dataset'  # folder containing geneix dataset
rows = 1 # num of crossbars in row dimension
cols = 1 # num of crossbars in col dimension
Gon = 1/100
Goff = 1/600
Vmax =0.25


# creating directory for dataset collection
if dataset:
    if not os.path.exists(direc):
        os.mkdir(direc)    

class NN_model(nn.Module):
    def __init__(self, N):
         super(NN_model, self).__init__()
         print ("WARNING: crossbar sizes with different row annd column dimension not supported.")
         self.fc1 = nn.Linear(N**2+N, 500)
         # self.bn1 = nn.BatchNorm1d(500)
         self.relu1 = nn.ReLU(inplace=True)
         self.do2 = nn.Dropout(0.5)
         self.fc3 = nn.Linear(500,N)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.do2(out)
        out = self.fc3(out)
        return out

xbmodel = NN_model(xbar_row_size)
xbmodel_weight_path = '../xb_models/XB_128_stream1slice207dropout50epochs.pth.tar' #'../xb_models/xbar_128x128_stream1_slice2_100k_600k_250mV.pth.tar'

#xbmodel = None
#xbmodel_weight_path = None

# Dump the current global configurations
def dump_config():
    param_dict = {'weight_bits':weight_bits, 'weight_bit_frac':weight_bit_frac, 'input_bits':input_bits, 'input_bit_frac':input_bit_frac, 
                  'xbar_row_size':xbar_row_size, 'xbar_col_size':xbar_col_size, 'tile_row':tile_row, 'tile_col':tile_col,
                  'bit_stream':bit_stream, 'bit_slice':bit_slice, 'adc_bit':adc_bit, 'acm_bits':acm_bits, 'acm_bit_frac':acm_bit_frac,
                   'mvm':mvm, 'non-ideality':non_ideality, '\nxbmodel':xbmodel, '\nxbmodel_weight_path':xbmodel_weight_path, 
                   'inmax_test':inmax_test, 'inmin_test':inmin_test}

    print('\n' + ' '*6 + "==> Functional simulator configurations:")
    for key, val in param_dict.items():
        t_str = ' '*10 + key + '=' + str(val)
        print (t_str)
    print('\n')