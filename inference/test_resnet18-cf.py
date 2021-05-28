#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:58:27 2021

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
frozen_mvm_models_dir = os.path.join(root_dir, "frozen_mvm_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, frozen_models_dir)
sys.path.insert(0, frozen_mvm_models_dir)
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

# User-defined packages
import frozen_models
import frozen_mvm_models
from utils.utils import accuracy, AverageMeter, save_checkpoint 
# User-defined packages
import models
from utils.data import get_dataset
from utils.preprocess import get_transform
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
for path, dirs, files in os.walk(frozen_models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()

#Create available models list
mvm_model_names = []
for path, dirs, files in os.walk(frozen_models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            mvm_model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
mvm_model_names.sort()


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

    end = time.time()

    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input_var = batch['data']
            target_var = batch['target'].long()
    
            input_var, target_var = input_var.to(device), target_var.to(device)

            if args.half:
                input_var = input_var.half()
            
            # compute output
            output = model(input_var)
            _, preds = torch.max(output, 1)
            for t, p in zip(batch['target'].view(-1), preds.view(-1)):
                confusion_matrix[t.cpu().detach().numpy(), p.cpu().detach().numpy()] += 1
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data, batch['data'].size(0))
            top1.update(prec1[0], batch['data'].size(0))
            top5.update(prec5[0], batch['data'].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                print('[{0}/{1}({2:.0f}%)]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(test_loader), 100. *float(batch_idx)/len(test_loader),
                    loss=losses, top1=top1, top5=top5))


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, top5=top5, loss=losses))
    print('Avg Loading time: {duration.avg:.4f} seconds'.format(duration=data_time))
    print('Avg Batch time: {duration.avg:.4f} seconds\n'.format(duration=batch_time))
    acc = top1.avg
    return acc, losses.avg, confusion_matrix
  
    
class SplitActivations_Dataset(Dataset):
    def __init__(self, args, datapath, tgtpath, train_len = True):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        self.args = args
        
    def __getitem__(self, index):
#        print('Index = ', index)
        if args.frozen_layers == 18:
            x = torch.load(os.path.join(self.datapath, 'act_fc'+'_'+str(index)+'.pth.tar'))
        else: 
            x = torch.load(os.path.join(self.datapath, 'act_relu'+str(self.args.frozen_layers)+'_'+str(index)+'.pth.tar'))
        
        y = torch.load(os.path.join(self.tgtpath, 'labels_'+str(index)+'.pth.tar'))
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
parser = argparse.ArgumentParser(description= 'Resnet18-CF-Inference')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
            help='dataset name')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18',
            choices=model_names,
            help='name of the model')

parser.add_argument('--load-dir', default='/home/nano01/a/esoufler/activations/x64-8b/',
            help='base path for loading activations')
parser.add_argument('--mode', default='rram',
                help='sram or rram')
parser.add_argument('--savedir', default='../results/discp_analysis/',
                help='base path for saving activations')
parser.add_argument('--pretrained', action='store', default='../pretrained_models/frozen/x64-8b/', #rram/cifar10/resnet18/freeze17_hp_best.pth.tar',
            help='the path to the ideal pretrained model')
parser.add_argument('--mvm', action='store_true', default=None,
                help='if running functional simulator backend')
parser.add_argument('--nideal', action='store_true', default=None,
            help='Add xbar non-idealities')

parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
            help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=300, type=int,
            metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--print-freq', '-p', default=5, type=int,
                metavar='N', help='print frequency (default: 5)')
parser.add_argument('--half', dest='half', action='store_true', default=None,
                    help='use half-precision(16-bit) ')

parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 0)')
parser.add_argument('--frozen-layers', default=1, type=int, help='number of frozen layers in the model')

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

# Check the savedir exists or not
save_dir = os.path.join(args.savedir, args.dataset, args.model)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#GPU init
os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
print('GPU Id(s) being used:', args.gpus)

#Create model
print('==> Building model for', args.model, '...')
if args.frozen_layers != 0 or args.frozen_layers != 18:
    if (args.model+'_freeze'+str(args.frozen_layers) in model_names):
        model = (__import__(args.model+'_freeze'+str(args.frozen_layers))) #import module using the string/variable_name
        
    if (args.model+'_freeze'+str(args.frozen_layers) in mvm_model_names):
        model_mvm = (__import__(args.model+'_freeze'+str(args.frozen_layers)+'_mvm')) #import module using the string/variable_name

    else:
        raise Exception(args.model+' is currently not supported')
else:
    model = (__import__(args.model))

print('==> Initializing model parameters ...')
weights_conv = []
weights_lin = []
bn_data = []
bn_bias = []
running_mean = []
running_var = []
num_batches = []

if not args.pretrained: #Initialize params with pretrained model
    raise Exception('Provide pretrained model for evaluation')
else:
    args.pretrained = os.path.join(args.pretrained, args.mode, args.dataset, args.model, 'freeze' + str(args.frozen_layers) + '_hp_best.pth.tar')
    print('==> Load pretrained model from', args.pretrained, '...')
    pretrained_model = torch.load(args.pretrained)
    best_acc = pretrained_model['best_acc']
    print('Pretrained model accuracy: {}'.format(best_acc))
    if args.dataset == 'cifar10':
        model = model.net(num_classes=10)
    elif args.dataset == 'cifar100':
        model = model.net(num_classes=100)
    
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

    if args.frozen_layers != 0:
        if args.dataset == 'cifar10':
            model_mvm = model_mvm.net(num_classes=10)
        elif args.dataset == 'cifar100':
            model_mvm = model_mvm.net(num_classes=100)
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
if args.mvm and args.frozen_layers != 0:
    cfg.mvm = True
    model = model_mvm

model.to(device)

if args.half:
    model = model.half()

model = torch.nn.DataParallel(model)
#%%
if args.frozen_layers != 0:
    if args.frozen_layers == 18:
        datapath_train = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'train', 'fc')
        datapath_test = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'test', 'fc')
    else:
        datapath_train = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'train', 'relu' + str(args.frozen_layers))
        datapath_test = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'test', 'relu' + str(args.frozen_layers))

    tgtpath_train = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'train', 'labels')
    tgtpath_test = os.path.join(args.load_dir, args.mode, 'one_batch', args.dataset, args.model, 'test', 'labels')


    train_data = SplitActivations_Dataset(args, datapath_train, tgtpath_train, train_len = True)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_data = SplitActivations_Dataset(args, datapath_test, tgtpath_test, train_len = False)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

else:
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
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_data = get_dataset(args.dataset, 'val', image_transforms['eval'], download=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()


import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if args.dataset == 'cifar10':
        tick_marks = np.arange(len(classes))
    elif args.dataset == 'cifar100':
        tick_marks = np.arange(0, len(classes), 5)

    plt.xticks(tick_marks, classes[tick_marks]-1, rotation=45)
    plt.yticks(tick_marks, classes[tick_marks]-1)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('../error_analysis/plots/' + args.dataset + '/' + args.mode + '_' + str(args.frozen_layers) + '.png')

if args.dataset == 'cifar10':
    nb_classes = 10
elif args.dataset == 'cifar100':
    nb_classes = 100

confusion_matrix = torch.zeros(nb_classes, nb_classes)

acc, loss, confusion_matrix = test(test_loader, model, criterion, device)
print('Prec@1 with ' + str(args.frozen_layers) + ' layers frozen = ', acc.item())




# classes = np.linspace(1,nb_classes, num=nb_classes)

# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot_confusion_matrix(np.array(confusion_matrix.cpu().detach().numpy(), dtype=np.int), classes)


