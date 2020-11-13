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

#Min
#savedir = '/home/min/a/akosta/Current_Projects/Hybrid_RRAM-SRAM/activations/multiple_batches/'

#Nano
#savedir= '/home/nano01/a/esoufler/activations/multiple_batches/'

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
def test(device):
    global best_acc
    flag = True
    training = False
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx,(data, target) in enumerate(testloader):
        data_var = data.to(device)
        target_var = target.to(device)
        
        output = model(data_var)
        loss= criterion(output, target_var)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        if flag == True:
            if batch_idx % 1 == 0:
                print('[{0}/{1}({2:.0f}%)]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       batch_idx, len(testloader), 100. *float(batch_idx)/len(testloader),
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
        out = output.detach()
        activation[name] = out
#        print('In: ' + name + ' ' +  str(activation[name].shape) + '  Out Device: ' + str(output.device)[-1])
        
    return hook1

def get_activation(name): #only works with 4 and 2 GPUs as of now
    def hook(module, input, output):
        
        out = output.detach()
        
        if len(args.gpus) == 7: #4 GPUS
            batch_split = int(args.batch_size/4) #0,1,2,3
            
            if str(output.device)[-1] == args.gpus[0]:
                act[name][0:batch_split] = out
            elif str(output.device)[-1] == args.gpus[2]:
                act[name][batch_split:2*batch_split] = out
            elif str(output.device)[-1] == args.gpus[4]:
                act[name][2*batch_split:3*batch_split] = out
            elif str(output.device)[-1] == args.gpus[6]:
                act[name][3*batch_split:4*batch_split] = out
            
        elif len(args.gpus) == 3: #2 GPUS
            batch_split = int(args.batch_size/2) #0,1 config ONLY
            
            if str(output.device)[-1] == args.gpus[0]:
                act[name][0:batch_split] = out
            elif str(output.device)[-1] == args.gpus[2]:
                act[name][batch_split:2*batch_split] = out
            
        elif len(args.gpus) == 1: # 1 GPU
            act[name] = out
        else:
            raise Exception('Odd multi-gpu numbers (3) not supported.')

#        print('In: ' + str(act[name].shape) + '  Out Device: ' + str(output.device)[-1])
    return hook

def reg_hook1(model):
    for name, module in model.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
#                print(name)
                module.register_forward_hook(get_activation1(name))
        
def reg_hook(model):
    for name, module in model.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                module.register_forward_hook(get_activation(name))

def save_activations(model, batch_idx, act, labels):
    global act_path
    for name, module in model.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                torch.save(act[name], os.path.join(act_path, name) + '/act_' + name + '_' + str(batch_idx) + '.pth.tar')
    torch.save(labels, act_path+'/labels/labels_' +str(batch_idx) + '.pth.tar')

    
#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=40, type=int, metavar='N', 
                help='mini-batch size (default: 40)')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                help='dataset name or folder')
    parser.add_argument('--savedir', default='/home/nano01/a/esoufler/activations/multiple_batches/',
                help='base path for saving activations')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18',
                choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet18fp_imnet.pth.tar',
                help='the path to the pretrained model')
    parser.add_argument('--mvm', action='store_true', default=False,
                help='if running functional simulator backend')
    parser.add_argument('--nideal', action='store_true', default=False,
                help='Add xbar non-idealities')
    parser.add_argument('--mode', default='test', 
                help='save activations for \'train\' or \'test\' sets')

    parser.add_argument('--start-batch', type=int, default=0,
                help='Start batch number')  
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 0,1,2,3)')
    parser.add_argument('-exp', '--experiment', default='128x128', metavar='N',
                help='experiment name')
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
      
    #print(model_mvm)

    print('==> Initializing model parameters ...')
    weights_conv = []
    weights_lin = []
    bn_data = []
    bn_bias = []
    running_mean = []
    running_var = []
    num_batches = []
    
    if not args.pretrained:
        raise Exception('Provide pretrained model for evalution')
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        print('Pretrained model accuracy: {}'.format(best_acc))
        base_model.load_state_dict(pretrained_model['state_dict'])
        for m in base_model.modules():
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
            
    i=j=k=0
    for m in model_mvm.modules():
        if isinstance(m, Conv2d_mvm):
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

    if args.mvm:
        cfg.mvm = True
        model = model_mvm
    
    model.to(device)#.half() # uncomment for FP16
    model = nn.DataParallel(model)
    
    
    
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

#    test(device)
#    exit(0)
    act_path = os.path.join(root_path, args.mode)
    
    if args.mode == 'train':
        dataloader = trainloader
    elif args.mode == 'test':
        dataloader = testloader
    else:
        raise Exception('Invalid save mode')
        
    for i in range(1, 18):
        os.makedirs(act_path + '/relu' + str(i), exist_ok=True)
    os.makedirs(act_path + '/fc', exist_ok=True)
    os.makedirs(act_path + '/labels', exist_ok=True)
    
#    for name, module in model.module.named_modules():
#        if 'relu' in name or 'fc' in name:
#            if 'xbmodel' not in name:
#                print(name)
    
    print("Saving activations to: {}".format(act_path))
    
    for batch_idx,(data, target) in enumerate(dataloader):
        base_time = time.time()
        reg_hook1(model)
        target = target.to(device)
        data_var = torch.autograd.Variable(data.to(device))
        target_var = torch.autograd.Variable(target.to(device))
        print('Dry run for computing activation sizes...')
        output = model(data_var)
#        print(activation['relu1'].shape())
        print('Dry run finished...')
        duration = time.time() - base_time
        print("Time taken: {}m {}secs".format(int(duration)//60, int(duration)%60))
        break
    
    for name, module in model.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                print(name + ': ' + str(activation[name].shape))

    act = {}
    for name, module in model.module.named_modules():
        if 'relu' in name and 'xbmodel' not in name:
            act[name] = torch.zeros([args.batch_size, activation[name].shape[1], activation[name].shape[2], activation[name].shape[3]])
            print(name + ': ' + str(act[name].shape))
        if 'fc' in name and 'xbmodel' not in name:
            act[name] = torch.zeros([args.batch_size, activation[name].shape[1]])
            print(name + ': ' + str(act[name].shape))
    labels = torch.zeros([args.batch_size])

    model.eval()
    for batch_idx,(data, target) in enumerate(dataloader):
        base_time = time.time()
        reg_hook(model)
        target = target.to(device)
        data_var = torch.autograd.Variable(data.to(device))
        target_var = torch.autograd.Variable(target.to(device))
    
        output = model(data_var)

        labels = target

        save_activations(model=model, batch_idx=batch_idx, act=act, labels=labels)
        duration = time.time() - base_time
        print("Batch IDx: {} \t Time taken: {}m {}secs".format(batch_idx, int(duration)//60, int(duration)%60))
    
    print("Done saving activations!")
    exit(0)