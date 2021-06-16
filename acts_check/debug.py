#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 22:42:24 2020

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
frozen_models_dir = os.path.join(root_dir, "frozen_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, frozen_models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

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

# User-defined packages
import models
import frozen_models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
import src.config as cfg

from src.pytorch_mvm_class_v3 import *

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

frozen_model_names = []
for path, dirs, files in os.walk(frozen_models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            frozen_model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
frozen_model_names.sort()



# Evaluate on a model
def test(test_loader, model, criterion, device, save=False):
    """
    Run evaluation
    """
    global suf
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    acts = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data_var = data.to(device)
            target_var = target.to(device)
            
            # compute output
            output_acts = model(data_var)
            acts.append(output_acts)
            labels.append(target_var)
            output = output_acts['out']
            
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            
            if save: 
                save_activations(output_acts, target_var, batch_idx, suf)
            elif batch_idx % 1 == 0:
                    print('[{0}/{1}({2:.0f}%)]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), 100. *float(batch_idx)/len(test_loader),
                        loss=losses, top1=top1, top5=top5))


    print('Baseline: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    acc = top1.avg
    return acts, labels, acc, losses.avg

def save_activations(acts, labels, batch_idx, suf):
    global act_path
    
    print('Batch IDx: ', batch_idx)
    for name in acts.keys():
        if 'relu' in name or 'fc' in name:
            if 'xbmodel' not in name:
                torch.save(acts[name], os.path.join(act_path, suf, name) + '/act_' + name + '_' + str(batch_idx) + '.pth.tar')
    torch.save(labels, os.path.join(act_path, suf, 'labels', 'labels_' +str(batch_idx) + '.pth.tar'))
    torch.save(acts['out'], os.path.join(act_path, suf, 'out' + '/act_out' + '_' + str(batch_idx) + '.pth.tar'))


def split_init(original_model, frozen_layers):
    M = (__import__(args.model+'_freeze'+str(frozen_layers)))
    M = M.net(num_classes=100).to(device)
    
    for name1, m1 in M.named_modules():
        for name2, m2 in original_model.module.named_modules():
            if name2 == name1:
                if 'resconv' in name1 or 'conv' in name1 or 'fc' in name1 or 'bn' in name1:
                    m1.load_state_dict(m2.state_dict())
    return M

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=250, type=int, metavar='N', 
            help='mini-batch size (default: 256)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
            help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
            choices=model_names,
            help='name of the model')
parser.add_argument('--savedir', type=str, default='./activations/',
            help='Save Directory')
parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar',
            help='the path to the pretrained model')
parser.add_argument('--mvm', action='store_true', default=True,
            help='if running functional simulator backend')
parser.add_argument('--nideal', action='store_true', default=True,
            help='Add xbar non-idealities')
parser.add_argument('--mode', default='test', 
            help='save activations for \'train\' or \'test\' sets')
parser.add_argument('--input_size', type=int, default=None,
            help='image input size')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
            help='number of data loading workers (default: 8)')
parser.add_argument('--gpus', default='0', help='gpus (default: 0,1,2,3)')
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

root_path = os.path.join(args.savedir, args.dataset)

if not os.path.exists(root_path):
    os.makedirs(root_path)

os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
print('GPU Id(s) being used:', args.gpus)

print('==> Building model and model_mvm for', args.model, '...')
if (args.model+'_direct' in model_names and args.model+'_mvm_direct' in model_names):
    model = (__import__(args.model+'_direct')) #import module using the string/variable_name
    model_mvm = (__import__(args.model+'_mvm_direct'))
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
    print('Pretrained model accuracy: {}'.format(best_acc))
    model.load_state_dict(pretrained_model['state_dict'])
    for m in model.modules():
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
for m in model_mvm.modules():
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

#%%

image_transforms = {
    'train':
        transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ]),
    'eval':
        transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
                ]),
    }

train_data = get_dataset(args.dataset, 'train', image_transforms['train'], download=True)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

test_data = get_dataset(args.dataset, 'val', image_transforms['eval'], download=True)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().to(device)

#    test(device)
#    exit(0)
act_path = root_path

if args.mode == 'train':
    dataloader = train_loader
elif args.mode == 'test':
    dataloader = test_loader
else:
    raise Exception('Invalid save mode')
    

act_path = '/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/cifar100/'
for i in range(1, 20):
    os.makedirs(act_path + '/sram_new/relu' + str(i), exist_ok=True)
os.makedirs(act_path + '/sram_new/fc', exist_ok=True)
os.makedirs(act_path + '/sram_new/labels', exist_ok=True)
os.makedirs(act_path + '/sram_new/out', exist_ok=True)
    
suf = 'sram_new'
print('Saving activations to: ' + str(act_path) + str(suf))

#%% TEST and SAVE_ACTIVATIONS
acts, labels, acc, loss = test(dataloader, model, criterion, device, save=True)

#%% TEST ON PART NETWORKS
for j in range(1,20,2):
    name = 'relu' + str(j)
    M = split_init(model, j).to(device)
    
    M.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for i in range(40):
        
        out = M(acts[i][name])
        
        prec1, prec5 = accuracy(out.data, labels[i].data, topk=(1, 5))
        
        top1.update(prec1[0], acts[i][name].size(0))
        top5.update(prec5[0], acts[i][name].size(0))
        
    #    if i % 1 == 0:
    #        print('[{0}/{1}({2:.0f}%)]\t'
    #            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #            i, 40, 100. *float(i)/40, top1=top1, top5=top5))
    
    
    print('Freeze {0}:  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(j, top1=top1, top5=top5))

del acts, labels, M

#%% SAVE SINGLE ACTIVATIONS

args.dataset = 'cifar100'
args.type = 'sram_new'
args.b_test = args.batch_size
################ SAVE THE LABELS (we do it only once)
base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'labels')
base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'labels')

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

print('Saving Test labels...')
for batch in range(num):
#    print(batch)
    filename = os.path.join(base_src_path, 'labels_' + str(batch) + '.pth.tar')
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.detach().cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = os.path.join(base_dst_path, 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

################ SAVE THE ACTIVATIONS 
print('Saving Test activations...')
for i in range(1,20,2):
    # testset
    base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'relu'+ str(i))
    base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'relu'+ str(i))
    
    print('relu' + str(i))

    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_test
    num = int(10000/batch_size)

    for batch in range(num):
#        print(batch)
        filename = os.path.join(base_src_path, 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar')
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

#################SAVE FC ACTIVATINS-----------------------------
# testset
base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'fc')
base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'fc')

print('fc')

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

for batch in range(num):
#        print(batch)
    filename = os.path.join(base_src_path, 'act_fc' +'_' + str(batch) + '.pth.tar')
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.detach().cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = os.path.join(base_dst_path, 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1
        
        
#%% ACTIVATIONS DATALOADER
class SplitActivations_Dataset(Dataset):
    def __init__(self, datapath, tgtpath, frozen_layers, train_len = True):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        self.frozen_layers = frozen_layers
        
    def __getitem__(self, index):
#        print('Index = ', index)
        if self.frozen_layers == 20:
            x = torch.load(os.path.join(self.datapath, 'act_fc'+'_'+str(index)+'.pth.tar'))
        else: 
            x = torch.load(os.path.join(self.datapath, 'act_relu'+str(self.frozen_layers)+'_'+str(index)+'.pth.tar'))
        
        y = torch.load(self.tgtpath+'/labels_'+str(index)+'.pth.tar')
        return {'data': x[0], 'target': y[0]}
    
    def __len__(self):
        if self.train_len == True:
            return 50000 
        else :
            return 10000

args.load_dir = '/home/nano01/a/esoufler/activations/error_analysis/one_batch/'
args.type = 'sram_new'
args.batch_size = 1000

for j in range(1,20,2):
    name = 'relu' + str(j)
    datapath_test = os.path.join(args.load_dir, args.dataset, args.type, name)
    tgtpath_test = os.path.join(args.load_dir, args.dataset, args.type, 'labels')
    
    
    test_data = SplitActivations_Dataset(datapath_test, tgtpath_test, frozen_layers=j, train_len=False)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    M = split_init(model, j).to(device)
    
    M.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            data = inputs['data']
            data_var = inputs['data'].to(device)
            target_var = inputs['target'].long().to(device)
            
            # compute output
            output = M(data_var)
            
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            
#            if save: 
#                save_activations(output_acts, target_var, batch_idx, suf)
            if batch_idx % 2 == 0:
                    print('[{0}/{1}({2:.0f}%)]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        batch_idx, len(test_loader), 100. *float(batch_idx)/len(test_loader),
                        loss=losses, top1=top1, top5=top5))


    print('Freeze {0}: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
          .format(j, top1=top1, top5=top5))


