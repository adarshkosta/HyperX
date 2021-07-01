#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:31:09 2020

@author: akosta
"""

import argparse
import shutil
import os
import sys
import time
import collections
import math

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "frozen_quantized_models")
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
from torch.autograd import Variable

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
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

parser = argparse.ArgumentParser(description='ResNet-20 CIFAR10 Training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                choices=model_names,
                help='name of the model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', 
                    help='weight decay (default: 5e-4)')

parser.add_argument('--tag', metavar='tag', default=None, type=str,
                    help='tag for save file name')

parser.add_argument('--milestones', default= [5,10,15,20], #[80,140,200], #[40, 80, 120, 150, 180], 
            help='Milestones for LR decay')
parser.add_argument('--gamma', default=0.2, type=float,
            help='learning rate decay')

parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', type=str, default=None,
                    help='pretrained model path')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--savedir', default='../pretrained_models/ideal/',
                help='base path for saving activations')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--gpus', default='0', help='gpus (default: 0)')

args = parser.parse_args()

print('\n' + ' '*6 + '==> Arguments:')
for k, v in vars(args).items():
    print(' '*10 + k + ': ' + str(v))

best_acc = 0
best_train_acc= 0

def train(trainloader, model, criterion, optimizer, epoch):
    global best_train_acc
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, target) in enumerate(trainloader):

        data = inputs.cuda()
        target = target.cuda()

        if args.half:
            data = data.half()

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
       
        loss.backward() 
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\tLR: {LR}'.format(
                       epoch, batch_idx, len(trainloader),
                       loss=losses, top1=top1, top5=top5, LR = optimizer.param_groups[0]['lr']))

    acc = top1.avg
    if acc > best_train_acc:
        best_train_acc = acc
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    print('Best Train Accuracy: {:.2f}%\n'.format(best_train_acc))
    return acc, losses.avg

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

            if batch_idx % 100 == 0:
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

if __name__=='__main__':
    # Check the savedir exists or not
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model for', args.model, '...')
    if (args.model in model_names):
        model = (__import__(args.model))
    else:
        raise Exception(args.model+'is currently not supported')
        
    if args.dataset == 'cifar10':
        model = model.net(num_classes=10)
    elif args.dataset == 'cifar100':
        model = model.net(num_classes=100)
    else:
      raise Exception(args.dataset + 'is currently not supported')
    
    if args.half:
        model = model.half()

    # print(model)

    # summary(model, (3,32,32))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 0
            best_acc = checkpoint['best_acc']
            best_train_acc = checkpoint['best_acc']
            # if args.dataset == 'cifar10':
            #     best_train_acc= 97.0 #80.0 #97.0
            # elif args.dataset == 'cifar100':
            #     best_train_acc= 78.0 #80.0 #97.0
            print('Resumed model accuracy: {}'.format(checkpoint['best_acc']))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint from {}".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = 0
            best_acc = 0
            best_train_acc= 0
            print('Pretrained model accuracy: {}'.format(checkpoint['best_acc']))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded pretrained model from {}".format(args.pretrained))
        else:
            print("=> no model found at '{}'".format(args.pretrained))
    else:
        print('Intializing model with normal distribution...')
        for m in model.modules():
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    if args.gpus and len(args.gpus) > 1:
        	model = torch.nn.DataParallel(model)

    model.to(device) 

    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    train_data = get_dataset(args.dataset, 'train', transform['train'],download=True)


    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_data = get_dataset(args.dataset, 'val', transform['eval'], download=True)
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    param_dict = dict(model.named_parameters())
    params = []

    base_lr = float(args.lr)
    wd = float(args.weight_decay)
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 
            'weight_decay': wd}]
    optimizer = optim.SGD(params, momentum = float(args.momentum), lr=args.lr,weight_decay=args.weight_decay, nesterov=True,dampening=0)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.milestones, 
                                                        gamma=args.gamma, 
                                                        last_epoch=args.start_epoch - 1)

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        [acc, loss] = test(testloader, model, criterion)
        exit(0)

    # evaluate on validation set
    [acc, loss] = test(testloader, model, criterion)
    print('Pretrained model accuracy: {}'.format(acc))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        [train_acc, train_loss] = train(trainloader, model, criterion, optimizer, epoch)

        #step learning rate
        lr_scheduler.step()
        
        # evaluate on validation set
        [acc, loss] = test(testloader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print('Best accuracy: {:.2f}%\n'.format(best_acc))

        if epoch > 0:
            if args.half:
                if args.tag:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_train_acc': best_train_acc,
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_half_' + str(args.tag))
                else:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_train_acc': best_train_acc,
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_half')
            else:
                if args.tag:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_train_acc': best_train_acc,
                        'best_acc': best_acc,
                    }, is_best, path=args.savedir, filename=args.model +'qfp_' + args.dataset + '_full_' + str(args.tag))
                else:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_train_acc': best_train_acc,
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_full')
