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
import pdb

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
frozen_models_dir = os.path.join(root_dir, "frozen_quantized_models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
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
# import frozen_models
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


# Training
def train(train_loader, model, criterion, optimizer, epoch, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    batch_freq = int(len(train_loader)/args.print_freq)
    
    for batch_idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input_var = batch['data']
        target_var = batch['target'].long()

        if args.half:
            input_var = input_var.half()

        input_var, target_var = input_var.to(device), target_var.to(device)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, batch['target'].to(device))[0]
        losses.update(loss.item(), batch['data'].size(0))
        top1.update(prec1.item(), batch['data'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % batch_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'], loss=losses, top1=top1, ))
    
    print('Total train loss: {loss.avg:.4f}\n'.format(loss=losses))

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
            data_var = batch['data'].cuda()
            target_var = batch['target'].long().cuda()
    
            if args.half:
                data_var = data_var.half()

            # data_var = data_var.to(device)
            # target_var = target_var.to(device)
            
            # compute output
            output = model(data_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data, batch['data'].size(0))
            top1.update(prec1[0], batch['data'].size(0))
            top5.update(prec5[0], batch['data'].size(0))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, top5=top5, loss=losses))
    acc = top1.avg

    return acc, losses.avg

class SplitActivations_Dataset(Dataset):
    def __init__(self, args, datapath, tgtpath, train_len=False):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        self.args = args
        
    def __getitem__(self, index):
#        print('Index = ', index)
        if args.frozen_layers == 20:
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
parser = argparse.ArgumentParser(description= ' Re-training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
            help='dataset name')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
            choices=model_names,
            help='name of the model')

parser.add_argument('--load-dir', default='/home/nano01/a/esoufler/activations/',
            help='base path for loading activations')
parser.add_argument('--savedir', default='../pretrained_models/frozen/',
                help='base path for saving activations')
parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20qfp_cifar100_i8b5f_w8b7f.pth.tar',
            help='the path to the ideal pretrained model')

parser.add_argument('--mode', default='rram',
                help='folder to load train activations from')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
            metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
            help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
            help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', default=0.1, type=float,
            help='learning rate decay')


parser.add_argument('--a_bit', default=7, type=int,
                metavar='N', help='activation total bits')
parser.add_argument('--af_bit', default=5, type=int,
                metavar='N', help='activation fractional bits')
parser.add_argument('--w_bit', default=7, type=int,
                metavar='N', help='weight total bits')
parser.add_argument('--wf_bit', default=7, type=int,
                metavar='N', help='weight fractional bits')


parser.add_argument('--milestones', default=[20,30,40,50], 
            help='Milestones for LR decay')

parser.add_argument('--exp', type=str, default='x128-8b',
            help='Crossbar size')
parser.add_argument('--loss', type=str, default='crossentropy', 
            help='Loss function to use')
parser.add_argument('--optim', type=str, default='sgd',
            help='Optimizer to use')

parser.add_argument('--print-freq', '-p', default=5, type=int,
                metavar='N', help='print frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False, 
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true', default=True,
                    help='use half-precision(16-bit) ')

parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--gpus', default='0', help='gpus (default: 0)')
parser.add_argument('--frozen-layers', default=1, type=int, help='number of frozen layers in the model')

args = parser.parse_args()

print_args(args)

args.mode_test = args.mode
args.mode_train = args.mode

args.savedir = os.path.join(args.savedir, args.exp)
args.load_dir = os.path.join(args.load_dir, args.exp)

# Check the savedir exists or not
save_dir = os.path.join(args.savedir, args.mode_test, args.dataset, args.model)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
    model = model.net(num_classes=10, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
elif args.dataset == 'cifar100':
    model = model.net(num_classes=100, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
else:
  raise Exception(args.dataset + 'is currently not supported')


# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print('Pretrained model accuracy: {}'.format(best_acc))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, checkpoint['epoch']))
    else:
        raise Exception(args.resume + ' does not exists')

else: #No model to resume from
    if args.pretrained: #Initialize params with pretrained model
        print('==> Initializing model with pre-trained parameters ...')
        original_model = (__import__(args.model))
        if args.dataset == 'cifar10':
            original_model = original_model.net(num_classes=10, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
        elif args.dataset == 'cifar100':
            original_model = original_model.net(num_classes=100, a_bit=args.a_bit, af_bit=args.af_bit, w_bit=args.w_bit, wf_bit=args.wf_bit)
        else:
            raise Exception(args.dataset + 'is currently not supported')

        print('==> Load pretrained model from', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        original_model.load_state_dict(pretrained_model['state_dict'])
        best_acc = pretrained_model['best_acc']
        print('Original model accuracy: {}'.format(best_acc))

        # for name1, m1 in model.named_modules():
        #     print(name1)

        # for name1, m1 in original_model.named_modules():
        #     print(name1)
        
        # pdb.set_trace()
        param_dict1 = model.state_dict()
        param_dict2 = original_model.state_dict()

        # print(param_dict1.keys())
        # print(param_dict2.keys())

        for key in param_dict1.keys():
            param_dict1[key] = param_dict2[key]
            # print(key)

        model.load_state_dict(param_dict1)

        # for name1, m1 in model.named_modules():
        #     for name2, m2 in original_model.named_modules():
        #         if name2 == name1:
        #             print(name1, name2)
        #             # if 'resconv' in name1 or 'conv' in name1 or 'fc' in name1 or 'bn' in name1 or 'fq' in name1:
        #             if '.' not in name1:
        #                 print(name1, name2)
        #                 m1.load_state_dict(m2.state_dict())

        # pdb.set_trace()
                        
    # else: #Initialize all params with normal distribution
    #     for m in model.modules():
    #         if isinstance(m, (nn.Conv2d, QConv2d)):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, (nn.Linear, QLinear)):
    #             stdv = 1. / math.sqrt(m.weight.data.size(1))
    #             m.weight.data.uniform_(-stdv, stdv)
    #             if m.bias is not None:
    #                m.bias.data.uniform_(-stdv, stdv)
 
model.to(device)
#%%
if args.frozen_layers == 20:
    datapath_train = os.path.join(args.load_dir, args.mode_train, 'one_batch', args.dataset, args.model, 'train', 'fc')
    datapath_test = os.path.join(args.load_dir, args.mode_test, 'one_batch', args.dataset, args.model, 'test', 'fc')
else:
    datapath_train = os.path.join(args.load_dir, args.mode_train, 'one_batch', args.dataset, args.model, 'train', 'relu' + str(args.frozen_layers))
    datapath_test = os.path.join(args.load_dir, args.mode_test, 'one_batch', args.dataset, args.model, 'test', 'relu' + str(args.frozen_layers))

tgtpath_train = os.path.join(args.load_dir, args.mode_train, 'one_batch', args.dataset, args.model, 'train', 'labels')
tgtpath_test = os.path.join(args.load_dir, args.mode_test, 'one_batch', args.dataset, args.model, 'test', 'labels')


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

if args.loss == 'nll':
    criterion = nn.NLLLoss()
elif args.loss == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
else:
    raise NotImplementedError

if args.half:
    model = model.half()
    criterion = criterion.half()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=True)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                            weight_decay=args.weight_decay)
else:
    raise NotImplementedError

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=args.milestones, 
                                            gamma=args.gamma, 
                                            last_epoch=args.start_epoch - 1)


# print(model)

if args.evaluate:
    acc, loss = test(test_loader, model, criterion, device)
    print('Prec@1 with ' + str(args.frozen_layers) + ' layers frozen = ', acc)
else:
    acc, loss = test(test_loader, model, criterion, device)
    print('Pre-trained Prec@1 with {} layers frozen: {} \t Loss: {}'.format(args.frozen_layers, acc.item(), loss.item()))
    print('\nStarting training on SRAM layers...')

    # pdb.set_trace()
    
    best_acc = 0
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)
        print('Train time: {}'.format(time.time()-end))

        # pdb.set_trace()
        # evaluate on validation set
        acc, loss = test(test_loader, model, criterion, device)
        
        # pdb.set_trace()

        lr_scheduler.step()
    
        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
            
        if args.half:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer,
            }, is_best, path=save_dir, filename='freeze' + str(args.frozen_layers) + '_hp')
        else:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer,
            }, is_best, path=save_dir, filename='freeze' + str(args.frozen_layers) + '_fp')
    
        print('Best acc: {:.3f}'.format(best_acc))
        print('-'*80)

        print('Test time: {}\n'.format(time.time()-end))
        end = time.time()