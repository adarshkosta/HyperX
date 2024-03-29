#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:36:33 2020

@author: akosta
"""
import os
import sys
import time
import collections

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

#%%
# Standard or Built-in packages
import numpy as np
import random
import argparse
import pdb
import math

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torchsummary import summary

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import models
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
for path, dirs, files in os.walk(models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

#%%
# Evaluate on a model
def test(test_loader, model, criterion, device):
    global best_acc

    model.eval()
    losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(test_loader):
        data_var = data.to(device)
        target_var = target.to(device)

        
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
  
# Intermediate feature maps
activation = {}
act = {}

def get_activation1(name):
    def hook1(module, input, output):
        output = output.detach()
        activation[name] = output
    return hook1

def reg_hook1(model):
    hook_handler1 = {}
    for name, module in model.module.named_modules():
        if 'InvertedResidual' in name or 'ConvBNReLU' in name or 'fc' in name:
            if 'xbmodel' not in name and '.' not in name:
                hook_handler1[name] = module.register_forward_hook(get_activation1(name))
    return hook_handler1

def unreg_hook(hook_handler):
    for name in hook_handler.keys():
        hook_handler[name].remove()

def get_activation(name): #only works with 4 and 2 GPUs as of now
    def hook(module, input, output):
        output = output.detach()
        if len(args.gpus) == 7: #4 GPUS
            batch_split = int(args.batch_size/4) #0,1,2,3
            
            if str(output.device)[-1] == args.gpus[0]:
                act[name][0:batch_split] = output
            elif str(output.device)[-1] == args.gpus[2]:
                act[name][batch_split:2*batch_split] = output
            elif str(output.device)[-1] == args.gpus[4]:
                act[name][2*batch_split:3*batch_split] = output
            elif str(output.device)[-1] == args.gpus[6]:
                act[name][3*batch_split:4*batch_split] = output
            
        elif len(args.gpus) == 3: #2 GPUS
            batch_split = int(args.batch_size/2) #0,1 config ONLY
            
            if str(output.device)[-1] == args.gpus[0]:
                act[name][0:batch_split] = output
            elif str(output.device)[-1] == args.gpus[2]:
                act[name][batch_split:2*batch_split] = output
            
        elif len(args.gpus) == 1: # 1 GPU
            act[name] = output
        else:
            raise Exception('Odd multi-gpu numbers (3) not supported.')
    return hook
        
def reg_hook(model):
    for name, module in model.module.named_modules():
        if 'InvertedResidual' in name or 'ConvBNReLU' in name or 'fc' in name:
            if 'xbmodel' not in name and '.' not in name:
                module.register_forward_hook(get_activation(name))

def save_activations(model, batch_idx, labels):
    global act_path, act
    for name, module in model.module.named_modules():
        if 'InvertedResidual' in name or 'ConvBNReLU' in name or 'fc' in name:
            if 'xbmodel' not in name and '.' not in name:
                torch.save(act[name], os.path.join(act_path, name) + '/act_' + name + '_' + str(batch_idx) + '.pth.tar')
    torch.save(labels, os.path.join(act_path, 'labels', 'labels_' +str(batch_idx) + '.pth.tar'))
    torch.save(act['out'], os.path.join(act_path, 'out', 'act_out_' +str(batch_idx) + '.pth.tar'))

    
#%%

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', 
            help='mini-batch size (default: 100)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
            help='dataset name or folder')
parser.add_argument('--savedir', default='/home/nano01/a/esoufler/activations/sram/multiple_batches/',
            help='base path for saving activations')
parser.add_argument('--model', '-a', metavar='MODEL', default='mobilenet',
            choices=model_names,
            help='name of the model')
parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/mobilenetv2fp_imnet.pth.tar',
            help='the path to the pretrained model')
parser.add_argument('--mvm', action='store_true', default=True,
            help='if running functional simulator backend')
parser.add_argument('--nideal', action='store_true', default=False,
            help='Add xbar non-idealities')
parser.add_argument('--mode', default='test', 
            help='save activations for \'train\' or \'test\' sets')

parser.add_argument('--input_size', type=int, default=None,
            help='image input size')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
            help='number of data loading workers (default: 8)')
parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 0,1,2,3)')
parser.add_argument('-exp', '--experiment', default='128x128', metavar='N',
            help='experiment name')
parser.add_argument('--batch-start', default=0, type=int, metavar='N', 
            help='Start batch')
args = parser.parse_args()

print('\n' + ' '*6 + '==> Arguments:')
for k, v in vars(args).items():
    print(' '*10 + k + ': ' + str(v))

if args.nideal:
    cfg.non_ideality = True
else:
    cfg.non_ideality = False
    
if args.mvm:
    cfg.mvm= True
else:
    cfg.mvm = False

#Using custom tiling
cfg.ifglobal_tile_col = False
cfg.ifglobal_tile_row = False
cfg.tile_row = 'custom'
cfg.tile_col = 'custom'


cfg.dump_config()

root_path = os.path.join(args.savedir, args.dataset, args.model)

os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
print(str(int((len(args.gpus)+1)/2)) + ' GPU devices being used. ID(s)', args.gpus)

print('==> Building model and model_mvm for', args.model, '...')
if (args.model in model_names and args.model+'_mvm' in model_names):
    base_model = (__import__(args.model))
    model = (__import__(args.model)) #import module using the string/variable_name
    model_mvm = (__import__(args.model+'_mvm'))
        
else:
    raise Exception(args.model+'is currently not supported')

base_model = model.net(num_classes=1000)

if args.dataset == 'cifar10':
    model = model.net(num_classes=10)
    model_mvm = model_mvm.net(num_classes=10)
elif args.dataset == 'cifar100':
    model = model.net(num_classes=100)
    model_mvm = model_mvm.net(num_classes=100)
else:
    raise Exception(args.dataset + 'is currently not supported')

print('==> Initializing model parameters ...')
weights_conv = []
weights_lin = []
bn_data = []
bn_bias = []
running_mean = []
running_var = []
num_batches = []

#Get params from pretrained model
if not args.pretrained:
    raise Exception('Provide pretrained model for evalution')
else:
    print('==> Load pretrained model form', args.pretrained, '...')
    pretrained_model = torch.load(args.pretrained, map_location=torch.device('cpu'))
    best_acc = pretrained_model['best_acc1']
    print('Pretrained model accuracy: {}'.format(best_acc))
    state_dict = pretrained_model['state_dict']

    #Remove module from loaded state_dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    base_model.load_state_dict(new_state_dict)

    for m in base_model.modules():
        if isinstance(m, nn.Conv2d):
            weights_conv.append(m.weight.data.clone())
        elif isinstance(m, nn.BatchNorm2d):
            bn_data.append(m.weight.data.clone())
            bn_bias.append(m.bias.data.clone())
            running_mean.append(m.running_mean.data.clone())
            running_var.append(m.running_var.data.clone())
            num_batches.append(m.num_batches_tracked.clone())
        elif isinstance(m, nn.Linear):
            weights_lin.append(m.weight.data.clone())

#Initialize Ideal model params
i=j=k=0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = weights_conv[i]
        i = i+1
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data = bn_data[j]
        m.bias.data = bn_bias[j]
        m.running_mean.data = running_mean[j]
        m.running_var.data = running_var[j]
        m.num_batches_tracked = num_batches[j]
        j = j+1
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.data.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

#Intitalize MVM model params
i=j=k=0
for m in model_mvm.modules():
    if isinstance(m, (Conv2d_mvm, nn.Conv2d)):
        m.weight.data = weights_conv[i]
        i = i+1
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data = bn_data[j]
        m.bias.data = bn_bias[j]
        m.running_mean.data = running_mean[j]
        m.running_var.data = running_var[j]
        m.num_batches_tracked = num_batches[j]
        j = j+1
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, Linear_mvm):
        stdv = 1. / math.sqrt(m.weight.data.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

if args.mvm:
    cfg.mvm = True

#Dataparallel models
model.to(device)#.half() # uncomment for FP16
model = nn.DataParallel(model)

model_mvm.to(device)#.half() # uncomment for FP16
model_mvm = nn.DataParallel(model_mvm)

#Input transforms
image_transforms = {
    'train':
        transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ]),
    'eval':
        transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ]),
    }

#Data and dataloaders
train_data = get_dataset(args.dataset, 'train', image_transforms['train'], download=True)
trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

test_data = get_dataset(args.dataset, 'val', image_transforms['eval'], download=True)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()

act_path = os.path.join(root_path, args.mode)

#Save train or test activations
if args.mode == 'train':
    dataloader = trainloader
    del testloader
elif args.mode == 'test':
    dataloader = testloader
    del trainloader
else:
    raise Exception('Invalid save mode')

#Create required dirs for saving activations    
for i in range(1, 18):
    os.makedirs(act_path + '/InvertedResidual' + str(i), exist_ok=True)
os.makedirs(act_path + '/ConvBNReLU1', exist_ok=True)
os.makedirs(act_path + '/ConvBNReLU2', exist_ok=True)
os.makedirs(act_path + '/fc', exist_ok=True)
os.makedirs(act_path + '/labels', exist_ok=True)
os.makedirs(act_path + '/out', exist_ok=True)

#Ideal model set to eval mode
model.eval()

#Set mvm model to eval
model_mvm.eval()

print("Saving activations to: {}".format(act_path))

#Dry run to compute activation sizes
data, target = next(iter(dataloader))
handler = reg_hook1(model)

data_var = data.to(device)
target_var = target.to(device)
print('Dry run for computing activation sizes...')
output = model(data_var)
print('Dry run finished...')

#Delete ideal model and its hook handler
# unreg_hook(handler)
# del model

#Single GPU activation shapes
print('\nFeature maps on single GPU')
for name, module in model_mvm.module.named_modules():
    if 'InvertedResidual' in name or 'ConvBNReLU' in name or 'fc' in name:
        if 'xbmodel' not in name and '.' not in name:
            print(name + ': ' + str(activation[name].shape))

#Multi-GPU activation shapes
print('\nFeature maps on multiple GPUs')
for name, module in model_mvm.module.named_modules():
    if 'InvertedResidual' in name and 'xbmodel' not in name and '.' not in name:
        act[name] = torch.zeros([args.batch_size, activation[name].shape[1], activation[name].shape[2], activation[name].shape[3]])
        print(name + ': ' + str(act[name].shape))
    elif 'ConvBNReLU' in name and 'xbmodel' not in name and '.' not in name:
        act[name] = torch.zeros([args.batch_size, activation[name].shape[1], activation[name].shape[2], activation[name].shape[3]])
        print(name + ': ' + str(act[name].shape))  
    elif 'fc' in name and 'xbmodel' not in name and '.' not in name:
        act[name] = torch.zeros([args.batch_size, activation[name].shape[1]])
        print(name + ': ' + str(act[name].shape))
    else:
        continue

print('Starting to save activations..')

#Iterate over dataloader and save activations
for batch_idx,(data, target) in enumerate(dataloader):
    if batch_idx >= args.batch_start:
        base_time = time.time()
        reg_hook(model_mvm)
        
        data_var = data.to(device)
        target_var = target.to(device)
        
        output = model_mvm(data_var)
        act['out'] = output

        save_activations(model=model_mvm, batch_idx=batch_idx, labels=target)
        
        duration = time.time() - base_time
        print("Batch IDx: {}\t Time taken: {}m {}secs".format(batch_idx, int(duration)//60, int(duration)%60))


print("Done saving activations!")
exit(0)