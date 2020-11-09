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

base_dir = '/home/nano01/a/esoufler/activations/error_analysis/one_batch/'
dataset = 'cifar100'
type = ['sram', 'rram']
layers = ['relu1', 'relu3', 'relu5', 'relu7', 'relu11', 'relu13', 'relu15', 'relu17', 'relu19'
          , 'fc']
idx = '0'

layer = 'relu1'

def load_activations(layer, name1, name2):
    path1 = os.path.join(base_dir, dataset, name1, layer, 'act_' + layer + '_' + idx + '.pth.tar')
    path2 = os.path.join(base_dir, dataset, name2, layer, 'act_' + layer + '_' + idx + '.pth.tar')
    
    acts1 = torch.load(path1)
    acts2 = torch.load(path2)
    
    return acts1, acts2

def compare_acts(layer, name1, name2, metric='mean'):
    acts1, acts2 = load_activations(layer, name1, name2)
    
    if metric == 'mean':
        mean1 = torch.mean(acts1)
        mean2 = torch.mean(acts2)
        
        err_mean = mean1 - mean2
    
        print('Mean1: {:.4f} \nMean2: {:.4f} \nMeandiff: {:.4f}'.format(mean1.item(), mean2.item(), err_mean.item()))
    
    elif metric == 'mse':
        sqerr = (acts1 - acts2)*(acts1 - acts2)
        
        mse = torch.mean(sqerr)
        
        print('MSE: {:.4f}'.format(mse.item()))
    
    elif metric == 'bool':
        check = torch.equal(acts1, acts2)
        diff = acts1 - acts2
        
        nzero = torch.nonzero(diff, as_tuple=True)
        
        
        if check == True:
            print(layer + ' activations are EQUAL!')
        else:
            print(layer + ' activations are NOT-EQUAL!!')
#            print(str(len(nzero)) + ' elements differ')
            print(nzero)
            
    

name1 = 'sram'
name2 = 'sram_direct'
for layer in layers:
    print('\nLayer: ', layer)
    compare_acts(layer, name1, name2, metric='bool')





