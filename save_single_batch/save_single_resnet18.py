#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:09:04 2020

@author: akosta
"""

import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-b_train', default=40, type=int,
                     metavar='N', help='mini-batch size (default: 40)')
parser.add_argument('-b_test', default=40, type=int,
                     metavar='N', help='mini-batch size (default: 40)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet18', help='name of the model')
args = parser.parse_args()

#%%
################ SAVE THE LABELS (we do it only once)
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'
batch_size = args.b_train
num = int(50000/batch_size)

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)


print('Saving Train labels...')
for batch in range(num):
#    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

#%%
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'
batch_size = args.b_test
num = int(10000/batch_size)

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)


print('Saving Test labels...')
for batch in range(num):
#    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

#%% 
################ SAVE THE ACTIVATIONS 
print('Saving Training and Test activations...')
for i in range(1,18,2):
    # trainset
    base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/relu'+ str(i) +'/'
    base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/relu'+ str(i) +'/'
    batch_size = args.b_train
    num = int(50000/batch_size)
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)

    
    print('relu'+str(i))
    for batch in range(num):
        filename = base_src_path + 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar' 
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = base_dst_path + 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

    # testset
    base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/relu'+ str(i) +'/'
    base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/relu'+ str(i) +'/'
    batch_size = args.b_test
    num = int(10000/batch_size)
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)


    for batch in range(num):
#        print(batch)
        filename = base_src_path + 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar' 
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = base_dst_path + 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

