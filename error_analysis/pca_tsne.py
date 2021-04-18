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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pkbar

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH

from utils.data import get_dataset

base_dir = '/home/nano01/a/esoufler/activations/x128/'
dataset = 'cifar10'
layers = ['relu3', 'relu5', 'relu7', 'relu9', 'relu11', 'relu13', 'relu15', 'relu17']
hw = 'rram_new'
mode = 'test'

save_path = os.path.join('plots', dataset, hw, mode)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if mode == 'test':
    nsamples = 10000
elif mode == 'train':
    nsamples = 50000

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

def load_activations(layer, mode):
    if layer == 'None':
        #Data and dataloaders
        data = get_dataset(dataset, mode, image_transforms[mode], download=True)
        loader = torch.utils.data.DataLoader(
        data,
        batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
        kbar = pkbar.Kbar(target=nsamples, width=20)

        for b_idx, (data, target) in enumerate(loader):
            act = data.numpy()
            tgt = target.numpy()

            try:
                acts = np.concatenate((acts, act))
                tgts = np.concatenate((tgts, tgt))
            except:
                acts = act
                tgts = tgt

            kbar.update(b_idx)

        return acts, tgts
    else:
        kbar = pkbar.Kbar(target=nsamples, width=20)
        for idx in range(nsamples):
            act_path = os.path.join(base_dir, hw, 'one_batch', dataset, 'resnet18', mode, layer, 'act_' + layer + '_' + str(idx) + '.pth.tar') 
            label_path = os.path.join(base_dir, hw, 'one_batch', dataset, 'resnet18', mode, 'labels', 'labels_' + str(idx) + '.pth.tar') 

            act = torch.load(act_path, map_location='cpu').cpu().detach().numpy() 
            tgt = torch.load(label_path, map_location='cpu').cpu().detach().numpy()

            try:
                acts = np.concatenate((acts, act))
                tgts = np.concatenate((tgts, tgt))
            except:
                acts = act
                tgts = tgt

            kbar.update(idx)

        tgts = tgts.astype('int64')
            
        return acts, tgts

def dim_reduction(acts, method='pca', n_components=2):
    pca = PCA(n_components=n_components, random_state=33)
    pca.fit(acts)
    # acts_train= pca.transform(acts_train)
    acts_test = pca.transform(acts)
    return acts, pca

def plot_reduction(X, Y, tgts, classes, layer):
    plt.figure(figsize=(20,15))
    color_map = plt.cm.get_cmap('Accent')
    
    #plot without labels (faster)
    plt.scatter(X, Y,c=tgts, cmap=color_map)

    #plot labels
    labels = np.array(classes)[tgts]
    class_num = set()
    for x1,x2,c,l in zip(X, Y, color_map(tgts), labels):
        if len(class_num)==10:
            break
        plt.scatter(x1,x2,c=[c],label=l)
        class_num.add(l)
        
    #remvoe duplicate labels    
    hand, labl = plt.gca().get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    plt.title('PCA-' + layer)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handout, lablout, fontsize=18)
    plt.savefig(os.path.join(save_path, 'pca-' + layer + '.png'))


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')    
print('Savedir: ', save_path)

for layer in layers:
    print('PCA at ' + layer + ':')
    # acts_train, tgts_train = load_activations(layer, mode='train')
    acts, tgts = load_activations(layer, mode='test')
    print(acts.shape)

    #flatten
    # acts_train = acts_train.reshape(-1, acts_train.shape[1]*acts_train.shape[2]*acts_train.shape[3])
    acts = acts.reshape(-1, acts.shape[1]*acts.shape[2]*acts.shape[3])

    acts, pca = dim_reduction(acts, method='pca', n_components=2)
    print(acts.shape)

    plot_reduction(acts[:,0], acts[:,1], tgts, classes, layer)

