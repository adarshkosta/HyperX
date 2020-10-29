#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:18:04 2020

@author: akosta
"""
import os
import sys

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
active_models_dir = os.path.join(root_dir, "active_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, active_models_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

#%%
# Standard or Built-in packages
import numpy as np
import random
import argparse
import pdb

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import active_models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
import src.config as cfg

from src.pytorch_mvm_class_v3 import *

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
for path, dirs, files in os.walk(active_models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

# Evaluate on a model
def test(test_loader, model, criterion, device):
    global best_acc
    flag = True
    training = False
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(test_loader):
        data_var = data.to(device)
        target_var = target.to(device)
        
        if args.half:
            data_var = data_var.half()
        
        output = model(data_var)
        loss= criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))


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
  
def print_args(args):
    print('\n' + ' '*6 + '==> Arguments:')
    for k, v in vars(args).items():
        print(' '*10 + k + ': ' + str(v))

parser = argparse.ArgumentParser(description= 'Test-active')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                     metavar='N', help='mini-batch size (default: 256 possible on 1 card)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
            help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
            choices=model_names,
            help='name of the model')
parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar',
    help='the path to the pretrained model')
parser.add_argument('--mvm', action='store_true', default=None,
            help='if running functional simulator backend')
parser.add_argument('--nideal', action='store_true', default=None,
            help='Add xbar non-idealities')

parser.add_argument('--input_size', type=int, default=None,
            help='image input size')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
            help='number of data loading workers (default: 8)')
parser.add_argument('--half', action='store_true', default=False,
            help='Use half-tensors')
parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 0)')
parser.add_argument('--active-layers', default=11, type=int, help='number of active layers from the left in the model')
args = parser.parse_args()

print_args(args)

if args.nideal:
  cfg.non_ideality = True
else:
  cfg.non_ideality = False
  
if args.mvm:
  cfg.mvm= True
else:
  cfg.mvm = False

cfg.dump_config()

os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
print('GPU Id(s) being used:', args.gpus)

print('==> Building model for', args.model, '...')
if (args.model in model_names and args.model+'_active'+str(args.active_layers) in model_names):
    model = (__import__(args.model+'_active'+str(args.active_layers)))
else:
    raise Exception(args.model+'is currently not supported')
    
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

    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []

    for m in original_model.modules():
        if isinstance(m, nn.Conv2d):
            weights_conv.append(m.weight.data.clone())
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            bn_data.append(m.weight.data.clone())
            bn_bias.append(m.bias.data.clone())
            running_mean.append(m.running_mean.data.clone())
            running_var.append(m.running_var.data.clone())
            num_batches.append(m.num_batches_tracked.clone())
        elif isinstance(m, nn.Linear):
            weights_lin.append(m.weight.data.clone())

    i=j=k=0
    for m in model.modules():
        if isinstance(m, (Conv2d_mvm, nn.Conv2d)):
            m.weight.data = weights_conv[i]
            i = i+1
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data = bn_data[j]
            m.bias.data = bn_bias[j]
            m.running_mean.data = running_mean[j]
            m.running_var.data = running_var[j]
            m.num_batches_tracked = num_batches[j]
            j = j+1
        # elif isinstance(m, Linear_mvm):
        #     m.weight.data = weights_lin[k]
        #     k=k+1
    model.fc.weight.data = original_model.fc.weight.data.clone()

    # for name1, m1 in model.named_modules():
    #     for name2, m2 in original_model.named_modules():
    #         if name2 == name1:
    #             if 'resconv' in name1 or 'conv' in name1 or 'fc' in name1 or 'bn' in name1:
    #                 m1.load_state_dict(m2.state_dict())
                    
else: #Initialize params with normal distribution
    raise Exception('Provide pretrained model for evalution')

if args.mvm:
    cfg.mvm = True

# Move required model to GPU (if applicable)
model.to(device)

if args.half:
    model.half() # FP16

model = torch.nn.DataParallel(model)

default_transform = {
    'train': get_transform(args.dataset,
                           input_size=args.input_size, augment=True),
    'eval': get_transform(args.dataset,
                          input_size=args.input_size, augment=False)
}
transform = getattr(model, 'input_transform', default_transform)

train_data = get_dataset(args.dataset, 'train', transform['train'])
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

test_data = get_dataset(args.dataset, 'val', transform['eval'], download=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()

if args.half:
    criterion.half()

test(test_loader, model, criterion, device)
exit(0)