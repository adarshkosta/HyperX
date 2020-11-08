#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:02:25 2020

@author: akosta
"""

import os
import sys
import time
import math

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
frozen_models_dir = os.path.join(root_dir, "frozen_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, frozen_models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

#%%
# Standard or Built-in packages
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import frozen_models
from utils.utils import accuracy, AverageMeter, save_checkpoint 
#from utils_bar import progress_bar

#Seeding
new_manual_seed = 0
torch.manual_seed(new_manual_seed)
torch.cuda.manual_seed_all(new_manual_seed)
np.random.seed(new_manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(new_manual_seed)
os.environ['PYTHONHASHSEED'] = str(new_manual_seed)

#Create available models list
model_names = []
for path, dirs, files in os.walk(frozen_models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

# Evaluate on a model
def test(test_loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_var = batch['data']
            target_var = batch['target'].type(torch.LongTensor)
    
            if args.half:
                input_var = input_var.half()
    
            input_var, target_var = input_var.to(device), target_var.to(device)
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data, batch['data'].size(0))
            top1.update(prec1[0], batch['data'].size(0))
            top5.update(prec5[0], batch['data'].size(0))
    
            if batch_idx % 1 == 0:
                    print('[{0}/{1}({2:.0f}%)]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), 100. *float(batch_idx)/len(test_loader),
                        loss=losses, top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    acc = top1.avg
    return acc, losses.avg
  
    
class SplitActivations_Dataset(Dataset):
    def __init__(self, datapath, tgtpath, train_len = True):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        
    def __getitem__(self, index):
#        print('Index = ', index)
        if args.frozen_layers == 20:
            x = torch.load(os.path.join(self.datapath, 'act_fc'+'_'+str(index)+'.pth.tar'))
        else: 
            x = torch.load(os.path.join(self.datapath, 'act_relu'+str(args.frozen_layers)+'_'+str(index)+'.pth.tar'))
        
        y = torch.load(self.tgtpath+'/labels_'+str(index)+'.pth.tar')
        return {'data': x[0], 'target': y[0]}
    
    def __len__(self):
        if self.train_len == True:
            return 50000 
        else :
            return 10000

def print_args(args):
    print('\n' + ' '*6 + '==> Arguments:')
    for k, v in vars(args).items():
        print(' '*10 + k + ': ' + str(v))


#Parse arguments
parser = argparse.ArgumentParser(description= 'Test-S')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
            help='dataset name')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
            choices=model_names,
            help='name of the model')
parser.add_argument('-b', '--batch-size', default=250, type=int,
            metavar='N', help='mini-batch size (default: 128)')


parser.add_argument('--load_dir', default='/home/nano01/a/esoufler/activations/error_analysis/one_batch/',
            help='base path for loading activations')
parser.add_argument('--type', default='sram_direct',
            help='type of activations to load')
parser.add_argument('--savedir', default='../pretrained_models/frozen/',
                help='base path for saving activations')

parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar',
            help='the path to the ideal pretrained model')

parser.add_argument('--input_size', type=int, default=None,
            help='image input size')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--half', dest='half', action='store_true', default=False,
            help='use half-precision(16-bit) ')
parser.add_argument('--gpus', default='0', help='gpus (default: 0)')
parser.add_argument('--frozen-layers', default=3, type=int, help='number of frozen layers in the model')

args = parser.parse_args()

print_args(args)

#GPU init
os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
print('GPU Id(s) being used:', args.gpus)

#Create model
print('==> Building model for', args.model, '...')
if (args.model+'_freeze'+str(args.frozen_layers) in model_names):
    model = (__import__(args.model+'_freeze'+str(args.frozen_layers))) #import module using the string/variable_name
else:
    raise Exception(args.model+' is currently not supported')

if args.dataset == 'cifar10':
    model = model.net(num_classes=10)
elif args.dataset == 'cifar100':
    model = model.net(num_classes=100)
else:
  raise Exception(args.dataset + 'is currently not supported')



if args.pretrained: #Initialize params with pretrained model
    print('==> Initializing model with pre-trained parameters ...')
    original_model = (__import__(args.model))
    if args.dataset == 'cifar10':
        original_model = original_model.net(num_classes=10)
    elif args.dataset == 'cifar100':
        original_model = original_model.net(num_classes=100)
    else:
        raise Exception(args.dataset + 'is currently not supported')

    print('==> Load pretrained model form', args.pretrained, '...')
    pretrained_model = torch.load(args.pretrained)
    original_model.load_state_dict(pretrained_model['state_dict'])
    best_acc = pretrained_model['best_acc']
    print('Original model accuracy: {}'.format(best_acc))
    
    for name1, m1 in model.named_modules():
        for name2, m2 in original_model.named_modules():
            if name2 == name1:
                if 'resconv' in name1 or 'conv' in name1 or 'fc' in name1 or 'bn' in name1:
                    m1.load_state_dict(m2.state_dict())
                    
else: #Initialize params with normal distribution
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.data.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

model.to(device)
#%%
if args.frozen_layers == 20:
    datapath_test = os.path.join(args.load_dir, args.dataset, args.type, 'fc')
else:
    datapath_test = os.path.join(args.load_dir, args.dataset, args.type, 'relu' + str(args.frozen_layers))

tgtpath_test = os.path.join(args.load_dir, args.dataset, args.type, 'labels')

#if args.frozen_layers == 20:
#    datapath_test = os.path.join(args.load_dir, args.dataset, args.model, 'test/fc')
#    datapath_train = os.path.join(args.load_dir, args.dataset, args.model, 'train/fc')
#else:
#    datapath_test = os.path.join(args.load_dir, args.dataset, args.model, 'test/relu' + str(args.frozen_layers))
#    datapath_train = os.path.join(args.load_dir, args.dataset, args.model, 'train/relu' + str(args.frozen_layers))
#
#tgtpath_test = os.path.join(args.load_dir, args.dataset, args.model, 'test/labels')
#tgtpath_train = os.path.join(args.load_dir, args.dataset, args.model, 'train/labels')


test_data = SplitActivations_Dataset(datapath_test, tgtpath_test, train_len=False)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()

if args.half:
    criterion.half()

acc, loss = test(test_loader, model, criterion, device)
print('Prec@1 with ' + str(args.frozen_layers) + ' layers frozen = ', acc.item(),
 '%\t Loss = ', loss.item())
