#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:42:24 2020

@author: akosta (original by Shubham Negi)
"""
import os
import sys
import time

#Min
#savedir = '/home/min/a/akosta/Current_Projects/Hybrid_RRAM-SRAM/activations/multiple_batches/'

#Nano
#savedir= '/home/nano01/a/esoufler/activations/multiple_batches/'

#Filepath handling
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *

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
        # if batch_idx == 10:
        #     break

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    acc = top1.avg
    return acc, losses.avg
  
# Intermediate feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def reg_hook(model):
    model.module.relu1.register_forward_hook(get_activation('relu1'))
    model.module.relu2.register_forward_hook(get_activation('relu2'))
    model.module.relu3.register_forward_hook(get_activation('relu3'))
    model.module.relu4.register_forward_hook(get_activation('relu4'))
    model.module.relu5.register_forward_hook(get_activation('relu5'))
    model.module.relu6.register_forward_hook(get_activation('relu6'))
    model.module.relu7.register_forward_hook(get_activation('relu7'))
    model.module.relu8.register_forward_hook(get_activation('relu8'))
    model.module.relu9.register_forward_hook(get_activation('relu9'))
    model.module.relu10.register_forward_hook(get_activation('relu10'))
    model.module.relu11.register_forward_hook(get_activation('relu11'))
    model.module.relu12.register_forward_hook(get_activation('relu12'))
    model.module.relu13.register_forward_hook(get_activation('relu13'))
    model.module.relu14.register_forward_hook(get_activation('relu14'))
    model.module.relu15.register_forward_hook(get_activation('relu15'))
    model.module.relu16.register_forward_hook(get_activation('relu16'))
    model.module.relu17.register_forward_hook(get_activation('relu17'))
    model.module.relu18.register_forward_hook(get_activation('relu18'))
    model.module.relu19.register_forward_hook(get_activation('relu19'))
    model.module.relu19.register_forward_hook(get_activation('fc'))

def save_activations(batch_idx):
    global act_path
    torch.save(act_relu1, act_path+'/relu1/act_relu1_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu2, act_path+'/relu2/act_relu2_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu3, act_path+'/relu3/act_relu3_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu4, act_path+'/relu4/act_relu4_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu5, act_path+'/relu5/act_relu5_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu6, act_path+'/relu6/act_relu6_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu7, act_path+'/relu7/act_relu7_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu8, act_path+'/relu8/act_relu8_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu9, act_path+'/relu9/act_relu9_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu10/act_relu10_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu11/act_relu11_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu12/act_relu12_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu13, act_path+'/relu13/act_relu13_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu14/act_relu14_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu15, act_path+'/relu15/act_relu15_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu16/act_relu16_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu17, act_path+'/relu17/act_relu17_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu18/act_relu18_' +str(batch_idx) + '.pth.tar')
    torch.save(act_relu11, act_path+'/relu19/act_relu19_' +str(batch_idx) + '.pth.tar')
    torch.save(act_fc, act_path+'/fc/act_fc_' +str(batch_idx) + '.pth.tar')
    torch.save(labels, act_path+'/labels/labels_' +str(batch_idx) + '.pth.tar')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                         metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--savedir', default='/home/nano01/a/esoufler/activations/multiple_batches/',
                help='base path for saving activations')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default=None,
        help='the path to the pretrained model')
    parser.add_argument('--mvm', action='store_true', default=None,
                help='if running functional simulator backend')
    parser.add_argument('--nideal', action='store_true', default=None,
                help='Add xbar non-idealities')
    parser.add_argument('--mode', default='test', help='save activations for \'train\' or \'test\' sets')
    
    parser.add_argument('--input_size', type=int, default=None,
                help='image input size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0', help='gpus (default: 4)')
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

    cfg.dump_config()
    
    root_path = os.path.join(args.savedir, args.dataset, args.model)
    
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model and model_mvm for', args.model, '...')
    if (args.model in model_names and args.model+'_mvm' in model_names):
        model = (__import__(args.model)) #import module using the string/variable_name
        model_mvm = (__import__(args.model+'_mvm'))
    else:
        raise Exception(args.model+'is currently not supported')
        
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
        model.load_state_dict(pretrained_model['state_dict'])
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
        if isinstance(m, (Conv2d_mvm, nn.Conv2d)):
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
        model = model_mvm
    
    model.to(device)#.half() # uncomment for FP16
    model = torch.nn.DataParallel(model)

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

    criterion = nn.CrossEntropyLoss()

#    test(device)
#    exit(0)
    activation = {}
    act_path = os.path.join(root_path, args.mode)
    
    if args.mode == 'train':
        dataloader = trainloader
    elif args.mode == 'test':
        dataloader = testloader
    else:
        raise Exception('Invalid save mode')
        
    os.makedirs(act_path + '/relu1', exist_ok=True)
    os.makedirs(act_path + '/relu2', exist_ok=True)
    os.makedirs(act_path + '/relu3', exist_ok=True)
    os.makedirs(act_path + '/relu4', exist_ok=True)
    os.makedirs(act_path + '/relu5', exist_ok=True)
    os.makedirs(act_path + '/relu6', exist_ok=True)
    os.makedirs(act_path + '/relu7', exist_ok=True)
    os.makedirs(act_path + '/relu8', exist_ok=True)
    os.makedirs(act_path + '/relu9', exist_ok=True)
    os.makedirs(act_path + '/relu10', exist_ok=True)
    os.makedirs(act_path + '/relu11', exist_ok=True)
    os.makedirs(act_path + '/relu12', exist_ok=True)
    os.makedirs(act_path + '/relu13', exist_ok=True)
    os.makedirs(act_path + '/relu14', exist_ok=True)
    os.makedirs(act_path + '/relu15', exist_ok=True)
    os.makedirs(act_path + '/relu16', exist_ok=True)
    os.makedirs(act_path + '/relu17', exist_ok=True)
    os.makedirs(act_path + '/relu18', exist_ok=True)
    os.makedirs(act_path + '/relu19', exist_ok=True)
    os.makedirs(act_path + '/fc', exist_ok=True)
    os.makedirs(act_path + '/labels', exist_ok=True)
    
    
    print("Saving activations to: {}".format(act_path))
    for batch_idx,(data, target) in enumerate(dataloader):
        reg_hook(model)
        target = target.to(device)
        data_var = torch.autograd.Variable(data.to(device))
        target_var = torch.autograd.Variable(target.to(device))
        print('Dry run for computing activation sizes...')
        output = model(data_var)
        
        break
    
    act_relu1 = torch.zeros([args.batch_size, activation['relu1'].shape[1], activation['relu1'].shape[2], activation['relu1'].shape[3]])
    act_relu2 = torch.zeros([args.batch_size, activation['relu2'].shape[1], activation['relu2'].shape[2], activation['relu2'].shape[3]])
    act_relu3 = torch.zeros([args.batch_size, activation['relu3'].shape[1], activation['relu3'].shape[2], activation['relu3'].shape[3]])
    act_relu4 = torch.zeros([args.batch_size, activation['relu4'].shape[1], activation['relu4'].shape[2], activation['relu4'].shape[3]])
    act_relu5 = torch.zeros([args.batch_size, activation['relu5'].shape[1], activation['relu5'].shape[2], activation['relu5'].shape[3]])
    act_relu6 = torch.zeros([args.batch_size, activation['relu6'].shape[1], activation['relu6'].shape[2], activation['relu6'].shape[3]])
    act_relu7 = torch.zeros([args.batch_size, activation['relu7'].shape[1], activation['relu7'].shape[2], activation['relu7'].shape[3]])
    act_relu8 = torch.zeros([args.batch_size, activation['relu8'].shape[1], activation['relu8'].shape[2], activation['relu8'].shape[3]])
    act_relu9 = torch.zeros([args.batch_size, activation['relu9'].shape[1], activation['relu9'].shape[2], activation['relu9'].shape[3]])
    act_relu10 = torch.zeros([args.batch_size, activation['relu10'].shape[1], activation['relu10'].shape[2], activation['relu10'].shape[3]])
    act_relu11 = torch.zeros([args.batch_size, activation['relu11'].shape[1], activation['relu11'].shape[2], activation['relu11'].shape[3]])
    act_relu12 = torch.zeros([args.batch_size, activation['relu12'].shape[1], activation['relu12'].shape[2], activation['relu12'].shape[3]])
    act_relu13 = torch.zeros([args.batch_size, activation['relu13'].shape[1], activation['relu13'].shape[2], activation['relu13'].shape[3]])
    act_relu14 = torch.zeros([args.batch_size, activation['relu14'].shape[1], activation['relu14'].shape[2], activation['relu14'].shape[3]])
    act_relu15 = torch.zeros([args.batch_size, activation['relu15'].shape[1], activation['relu15'].shape[2], activation['relu15'].shape[3]])
    act_relu16 = torch.zeros([args.batch_size, activation['relu16'].shape[1], activation['relu16'].shape[2], activation['relu16'].shape[3]])
    act_relu17 = torch.zeros([args.batch_size, activation['relu17'].shape[1], activation['relu17'].shape[2], activation['relu17'].shape[3]])
    act_relu18 = torch.zeros([args.batch_size, activation['relu18'].shape[1], activation['relu18'].shape[2], activation['relu18'].shape[3]])
    act_relu19 = torch.zeros([args.batch_size, activation['relu19'].shape[1], activation['relu19'].shape[2], activation['relu19'].shape[3]])
    act_fc = torch.zeros([args.batch_size, activation['fc'].shape[1], activation['relu17'].shape[2], activation['relu17'].shape[3]])
    labels = torch.zeros([args.batch_size])

    for batch_idx,(data, target) in enumerate(dataloader):
        base_time = time.time()
        reg_hook(model)
        target = target.to(device)
        data_var = torch.autograd.Variable(data.to(device))
        target_var = torch.autograd.Variable(target.to(device))
    
        output = model(data_var)
        
        act_relu1 = activation['relu1']
        act_relu2 = activation['relu2']
        act_relu3 = activation['relu3']
        act_relu4= activation['relu4']
        act_relu5 = activation['relu5']
        act_relu6 = activation['relu6']
        act_relu7 = activation['relu7']
        act_relu8 = activation['relu8']
        act_relu9 = activation['relu9']
        act_relu10 = activation['relu10']
        act_relu11 = activation['relu11']
        act_relu12 = activation['relu12']
        act_relu13 = activation['relu13']
        act_relu14 = activation['relu14']
        act_relu15 = activation['relu15']
        act_relu16 = activation['relu16']
        act_relu17 = activation['relu17']
        act_relu18 = activation['relu18']
        act_relu19 = activation['relu19']
        act_fc = activation['fc']
        labels = target
        
        duration = time.time() - base_time
#        print("Batch IDx: {}".format(batch_idx))
        print("Batch IDx: {} \t Time taken: {}m {}secs".format(batch_idx, int(duration)//60, int(duration)%60))    
        save_activations(batch_idx=batch_idx)
    