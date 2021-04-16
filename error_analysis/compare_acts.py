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

import matplotlib.pyplot as plt
import pandas as pd

base_dir = '/home/nano01/a/esoufler/activations/x128/'
dataset = 'cifar100'
layers = ['relu1', 'relu3', 'relu5', 'relu7', 'relu9', 'relu11', 'relu13', 'relu15', 'relu17']
# idx = '1'


save_path = './plots/' + dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)

mode = 'test'

if mode == 'test':
    nsamples = 10000
elif mode == 'train':
    nsamples = 50

def load_activations(layer, idx):
    path1 = os.path.join(base_dir, 'sram', 'one_batch', dataset, 'resnet18', mode, layer, 'act_' + layer + '_' + str(idx) + '.pth.tar')
    path2 = os.path.join(base_dir, 'rram', 'one_batch', dataset, 'resnet18', mode, layer, 'act_' + layer + '_' + str(idx) + '.pth.tar')
    
    acts1 = torch.load(path1)
    acts2 = torch.load(path2)
    
    return acts1, acts2

def compare_acts(layer, metric='mean'):
    err_mean = torch.zeros(1)
    err_mse = torch.zeros(1)

    for idx in range(nsamples):
        acts1, acts2 = load_activations(layer, idx)
        if metric == 'mean':
            mean1 = torch.mean(acts1)
            mean2 = torch.mean(acts2)
            
            err_mean = torch.cat((err_mean, torch.mean(acts1-acts2).unsqueeze(0)), dim=0) #mean1 - mean2
        
            # print('Mean1: {:.4f} \nMean2: {:.4f} \nMeandiff: {:.4f}'.format(mean1.item(), mean2.item(), err_mean.item()))

        elif metric == 'mse':
            sqerr = (acts1 - acts2)*(acts1 - acts2)
            # print(torch.mean(sqerr).unsqueeze(0))
            err_mse = torch.cat((err_mse, torch.mean(sqerr).unsqueeze(0)), dim=0)
    
    if metric == 'mean':
        return err_mean.mean().item()
    elif metric == 'mse':
        return err_mse.mean().item()


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
    
def visualize_acts(layer):
    acts1, acts2 = load_activations(layer)

    # print(acts1.shape, acts2.shape)

    acts1 = acts1.squeeze()
    acts2 = acts2.squeeze()
    print(layer, ':', acts1.shape)

    acts = torch.stack((acts1, acts2))
    num = 10

    plot(acts, num)
    

# for layer in layers:
#     visualize_acts(layer)


mean_diff = []
mse = []
l = []

i=0
for layer in layers:
    i = i+1
    mse.append(compare_acts(layer, metric='mse'))
    mean_diff.append(compare_acts(layer, metric='mean'))
    l.append(int(layer[4:]))

mse = np.array(mse)
mean_diff = np.array(mean_diff)
l= np.array(l)

metrics = np.stack((l, mse, mean_diff), axis=-1)
print(metrics)

np.savetxt(os.path.join(save_path, 'metrics_' + dataset + '.csv'), metrics, delimiter=',')


plt.plot(l, mse)
plt.plot(l, mean_diff)
plt.title(dataset)
plt.show()