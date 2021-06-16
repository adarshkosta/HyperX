#!/bin/bash

python save_mvm_resnet20.py --mvm --nideal --fsmodel -b1000 --dataset='cifar10' --mode='train' --pretrained='../pretrained_models/ideal/resnet20fxp_cifar10_i8b4f_w8b6f.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x128-8b/rram/multiple_batches/' 

python save_mvm_resnet20.py --mvm --nideal --fsmodel -b1000 --dataset='cifar10' --mode='test' --pretrained='../pretrained_models/ideal/resnet20fxp_cifar10_i8b4f_w8b6f.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x128-8b/rram/multiple_batches/' 

