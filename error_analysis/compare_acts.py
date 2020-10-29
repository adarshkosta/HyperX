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

base_dir = './activations'
dataset = 'cifar100'
type = ['sram', 'rram']
layers = ['relu1', 'relu3', 'relu5', 'relu7', 'relu11', 'relu13', 'relu15', 'relu17', 'relu19'
          , 'fc']
idx = '0'

layer = 'relu19'

def load_activations(layer):
    sram_path = os.path.join(base_dir, dataset, 'sram', 'act_' + layer + '_' + idx + '.pth.tar')
    rram_path = os.path.join(base_dir, dataset, 'rram', 'act_' + layer + '_' + idx + '.pth.tar')
    
    sram_acts = torch.load(sram_path)
    rram_acts = torch.load(rram_path)
    
    return sram_acts, rram_acts

def compare_acts(layer, metric='mean'):
    sram_acts, rram_acts = load_activations(layer)
    
    if metric == 'mean':
        sram_mean = torch.mean(sram_acts)
        rram_mean = torch.mean(rram_acts)
        
        err_mean = sram_mean - rram_mean
    
        print('SRAM mean: {:.4f} \nRRAM mean: {:.4f} \nMeandiff: {:.4f}'.format(sram_mean.item(), rram_mean.item(), err_mean.item()))
    
    elif metric == 'mse':
        sqerr = (sram_acts - rram_acts)*(sram_acts - rram_acts)
        
        mse = torch.mean(sqerr)
        
        print('MSE: {:.4f}'.format(mse.item()))
        
    


for layer in layers:
    print('\nLayer: ', layer)
    compare_acts(layer, metric='mean')
    compare_acts(layer, metric='mse')



