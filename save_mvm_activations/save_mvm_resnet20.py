#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:42:24 2020

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

from frozen_quantized_models.quant_dorefa import *

#Seeding
def reset_seed():
    new_manual_seed = 0
    torch.manual_seed(new_manual_seed)
    torch.cuda.manual_seed_all(new_manual_seed)
    np.random.seed(new_manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(new_manual_seed)
    os.environ['PYTHONHASHSEED'] = str(new_manual_seed)

reset_seed()

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
def test(testloader, model, criterion):
    global best_acc
    flag = False
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(testloader):
            target = target.cuda()
            data_var = data.cuda()
            target_var = target.cuda()

            if args.half:
                data_var = data_var.half()

                                        
            output = model(data_var)
            loss= criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            if batch_idx % 1 == 0:
                print('Test: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(testloader),
                    loss=losses, top1=top1, top5=top5))

    acc = top1.avg
    
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

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
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
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
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                module.register_forward_hook(get_activation(name))

def save_activations(model, batch_idx, act, labels):
    global act_path
    for name, module in model.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                torch.save(act[name], os.path.join(act_path, name) + '/act_' + name + '_' + str(batch_idx) + '.pth.tar')
    torch.save(labels, os.path.join(act_path, 'labels', 'labels_' +str(batch_idx) + '.pth.tar'))
    torch.save(act['out'], os.path.join(act_path, 'out', 'act_out_' +str(batch_idx) + '.pth.tar'))


#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=1000, type=int, metavar='N', 
                help='mini-batch size (default: 1000)')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--savedir', default='/home/nano01/a/esoufler/activations/x64-8b/rram/multiple_batches/',
                help='base path for saving activations')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20fp_cifar10.pth.tar',
                help='the path to the pretrained model')
    parser.add_argument('--mvm', action='store_true', default=None,
                help='if running functional simulator backend')
    parser.add_argument('--nideal', action='store_true', default=None,
                help='Add xbar non-idealities')
    parser.add_argument('--mode', default='test', 
                help='save activations for \'train\' or \'test\' sets')

    parser.add_argument('--quantize-model', action='store_true', default=None,
                help='quantize model weights') 
    parser.add_argument('--fsmodel', action='store_true', default=None,
                help='Ideal model or FS model')

    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 0,1,2,3)')
    parser.add_argument('-exp', '--experiment', default='x128', metavar='N',
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

    cfg.dump_config()
    
    root_path = os.path.join(args.savedir, args.dataset, args.model)
    
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model and model_mvm for', args.model, '...')
    if (args.model in model_names and args.model+'_mvm' in model_names):
        model = (__import__(args.model)) #import module using the string/variable_name
        model_mvm = (__import__(args.model+'_mvm'))
    else:
        raise Exception(args.model+'is currently not supported')

    if args.dataset == 'cifar10':
        model = model.net(num_classes=10)#, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
        model_mvm = model_mvm.net(num_classes=10)#, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
    elif args.dataset == 'cifar100':
        model = model.net(num_classes=100)#, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
        model_mvm = model_mvm.net(num_classes=100)#, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
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

        if args.fsmodel:
            state_dict = {key:val for key, val in pretrained_model['state_dict'].items() if 'xbmodel' not in key}
        else:
            state_dict = pretrained_model['state_dict']

        model.load_state_dict(state_dict)

        #Quantize weights
        if args.quantize_model:
            if args.dataset == 'cifar10':
                wf_bit = 6
            elif args.dataset == 'cifar100':
                wf_bit = 7
            wt_quant = weight_quantize_fn(w_bit=7, wf_bit=wf_bit) 
            for name, m in model.named_modules():
                # print(name)
                if 'fc' in name and 'quantize_fn' not in name:
                    m.weight.data = wt_quant(m.weight.data)
                elif 'resconv' in name:
                    if '.0' in name and 'quantize_fn' not in name:
                        m.weight.data = wt_quant(m.weight.data)
                elif 'conv' in name and 'quantize_fn' not in name:
                    m.weight.data = wt_quant(m.weight.data)


        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                weights_conv.append(m.weight.data.clone())
                #print(m.weight.data)
                #raw_input()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_data.append(m.weight.data.clone())
                bn_bias.append(m.bias.data.clone())
                running_mean.append(m.running_mean.data.clone())
                running_var.append(m.running_var.data.clone())
                num_batches.append(m.num_batches_tracked.clone())
            elif isinstance(m, nn.Linear):
                weights_lin.append(m.weight.data.clone())
    
        i=j=k=0
        for m in model_mvm.modules():
            if isinstance(m, (Conv2d_mvm)):
                m.weight.data = weights_conv[i]
                i = i+1
            #print(m.weight.data)
            #raw_input()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data = bn_data[j]
                m.bias.data = bn_bias[j]
                m.running_mean.data = running_mean[j]
                m.running_var.data = running_var[j]
                m.num_batches_tracked = num_batches[j]
                j = j+1
            elif isinstance(m, Linear_mvm):
                m.weight.data = weights_lin[k]
                k=k+1

    # Move required model to GPU (if applicable)
    if args.mvm:
        cfg.mvm = True
    
    model.to(device)#.half() # uncomment for FP16
    model = nn.DataParallel(model)

    model_mvm.to(device)#.half() # uncomment for FP16
    model_mvm = nn.DataParallel(model_mvm)

    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_data = get_dataset(args.dataset, 'val', transform['eval'], download=True)
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    act_path = os.path.join(root_path, args.mode)
    
    if args.mode == 'train':
        dataloader = trainloader
    elif args.mode == 'test':
        dataloader = testloader
    else:
        raise Exception('Invalid save mode')
        
    for i in range(1, 20):
        os.makedirs(act_path + '/relu' + str(i), exist_ok=True)
    os.makedirs(act_path + '/fc', exist_ok=True)
    os.makedirs(act_path + '/labels', exist_ok=True)
    os.makedirs(act_path + '/out', exist_ok=True)
    
    #Ideal model set to eval mode
    model.eval()

    #Set mvm model to eval
    model_mvm.eval()
    
    print("Saving activations to: {}".format(act_path))
    data, target = next(iter(dataloader))
    handler = reg_hook1(model)
    
    data_var = data.to(device)
    target_var = target.to(device)
    print('Dry run for computing activation sizes...')
    output = model(data_var)
    print('Dry run finished...')
    
    unreg_hook(handler)
    del model
    
    #Single GPU activation shapes
    for name, module in model_mvm.module.named_modules():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                print(name + ': ' + str(activation[name].shape))

    #Multi-GPU activation shapes
    for name, module in model_mvm.module.named_modules():
        if 'relu' in name and 'xbmodel' not in name:
            act[name] = torch.zeros([args.batch_size, activation[name].shape[1], activation[name].shape[2], activation[name].shape[3]])
            print(name + ': ' + str(act[name].shape))
        elif 'fc' in name and 'xbmodel' not in name:
            act[name] = torch.zeros([args.batch_size, activation[name].shape[1]])
            print(name + ': ' + str(act[name].shape))
        else:
            continue

    

    print('Starting to save activations..')

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #Iterate over dataloader and save activations

    for batch_idx,(data, target) in enumerate(dataloader):

        if batch_idx >= args.batch_start:
            base_time = time.time()
            reg_hook(model_mvm)
            
            data_var = data.cuda()
            target_var = target.cuda()
            
            output = model_mvm(data_var)
            act['out'] = output

            loss= criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            save_activations(model=model_mvm, batch_idx=batch_idx, act=act, labels=target)
            
            duration = time.time() - base_time
            
            print('Batch: [{0}/{1}]\t'
                'Time {2}m {3}s\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(dataloader), int(duration)//60, int(duration)%60,
                loss=losses, top1=top1, top5=top5))

            
            # print("Batch IDx: {} \t Time taken: {}m {}secs".format(batch_idx, int(duration)//60, int(duration)%60))
    
    print("Done saving activations!")
    exit(0)