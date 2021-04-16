#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:27:15 2020

@author: akosta
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH

from utils.data import get_dataset

base_dir = '/home/nano01/a/esoufler/activations/x128/'
dataset = 'cifar10'
layers = ['relu1', 'relu3', 'relu5', 'relu7', 'relu9', 'relu11', 'relu13', 'relu15', 'relu17']
hw = 'rram'


save_path = './plots/' + dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)

mode = 'test'

if mode == 'test':
    nsamples = 10
elif mode == 'train':
    nsamples = 50

image_transforms = {
        'train':
            transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                    ]),
        'test':
            transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                    ]),
        }

def load_activations(layer, idx):
    if layer == None:
        #Data and dataloaders
        data = get_dataset(dataset, mode, image_transforms[mode], download=True)
        loader = torch.utils.data.DataLoader(
        data,
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)

        for bidx, (data, target) in enumerate(loader):
            acts

        return data
    else:
        path = os.path.join(base_dir, hw, 'one_batch', dataset, 'resnet18', mode, layer, 'act_' + layer + '_' + str(idx) + '.pth.tar')  
        acts = torch.load(path) 
        return acts


def plot(acts, num=10):
    fig, axarr = plt.subplots(acts.size(0)+1, num, figsize=(20,2*(acts.size(0)+1)))
    cmap='gray'

    for i in range(acts.size(0)):
        for j in range(num):
           im = axarr[i,j].imshow(acts[i,j], cmap=cmap)

    for j in range(num):
        im = axarr[2,j].imshow(acts[0,j]-acts[1,j], cmap=cmap)    

    axarr[0,0].set_ylabel('SRAM')
    axarr[1,0].set_ylabel('RRAM')
    axarr[2,0].set_ylabel('DIFF')
    fig.suptitle(dataset + '-' + layer)

    plt.savefig(os.path.join(save_path, dataset + '-' + layer + '.jpg'))
    
data = load_activations(None, 0)
print(data.shape)