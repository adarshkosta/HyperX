#!/bin/bash

#python save_mvm_resnet20.py --mvm --nideal -b2000 --dataset='cifar100' --mode='train' --pretrained='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/rram/multiple_batches/' 

#python save_mvm_resnet20.py --mvm --nideal -b2000 --dataset='cifar100' --mode='test' --pretrained='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/rram/multiple_batches/' 

python save_mvm_resnet20.py --mvm --nideal --fsmodel -b1000 --dataset='cifar100' --mode='train' --pretrained='../pretrained_models/ideal/resnet20fxp_cifar100_i8b5f_w8b7f.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x128-8b/rram/multiple_batches/' 

python save_mvm_resnet20.py --mvm --nideal --fsmodel -b1000 --dataset='cifar100' --mode='test' --pretrained='../pretrained_models/ideal/resnet20fxp_cifar100_i8b5f_w8b7f.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x128-8b/rram/multiple_batches/' 
