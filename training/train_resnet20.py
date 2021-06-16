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


#Min
#savedir = '/home/min/a/akosta/Current_Projects/Hybrid_RRAM-SRAM/activations/multiple_batches/'

#Nano
#savedir= '/home/nano01/a/esoufler/activations/multiple_batches/'

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
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

parser.add_argument('--tag', metavar='tag', default=None, type=str,
                    help='use first and last layers with 16b precision and assign a tag to sim')

parser.add_argument('--milestones', default=[80, 120, 160], 
            help='Milestones for LR decay')
parser.add_argument('--gamma', default=0.1, type=float,
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

def main():
    global args, best_acc

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

    model.to(device)#.half() # uncomment for FP16

    # summary(model, (3,32,32))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 0
            best_acc = 0
            best_train_acc= 0
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
    # else:
    #     print("=> Intializing model '{}'".format(args.pretrained))
    #     for name, m in model.named_modules():
    #         if 'conv' in name:
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif 'bn' in name:
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif 'fc' in name:
    #             stdv = 1. / math.sqrt(m.weight.data.size(1))
    #             m.weight.data.uniform_(-stdv, stdv)
    #             if m.bias is not None:
    #                m.bias.data.uniform_(-stdv, stdv)

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
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_data = get_dataset(args.dataset, 'val', transform['eval'], download=True)
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.half:
        criterion = criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.milestones, 
                                                        gamma=args.gamma, 
                                                        last_epoch=args.start_epoch - 1)



    if args.evaluate:
        validate(testloader, model, criterion)
        return

    # evaluate on validation set
    [acc, loss] = validate(testloader, model, criterion)
    print('Pretrained model accuracy: {}'.format(acc))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        [train_acc, train_loss] = train(trainloader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        [acc, loss] = validate(testloader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print('Best accuracy: {}\n'.format(best_acc))

        if epoch > 0 and epoch % args.save_every == 0:
            if args.half:
                if args.tag:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_half_' + str(args.tag))
                else:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_half')
            else:
                if args.tag:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, is_best, path=args.savedir, filename=args.model +'qfp_' + args.dataset + '_full_' + str(args.tag))
                else:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                    }, is_best, path= args.savedir, filename=args.model +'qfp_' + args.dataset + '_full')

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    global best_train_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # prec1 = accuracy(output.data, target)[0]
        # losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {3}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), optimizer.param_groups[0]['lr'],
                      loss=losses, top1=top1))

    acc = top1.avg
    if acc > best_train_acc:
        best_train_acc = acc
    print('Train Epoch: {}\t({:.2f}%)]\tLoss: {:.6f}\n'.format(
                epoch,
                acc, losses.avg))

    print('Best Accuracy: {:.2f}%\n'.format(best_train_acc))
    return acc, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    global best_acc
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target#.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # output = output.float()
            # loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target)[0]
            # losses.update(loss.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(val_loader), loss=losses,
                            top1=top1))


    return top1.avg, losses.avg

#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#    """s
#    Save the training model
#    """
#    torch.save(state, filename)

if __name__ == '__main__':
    main()
