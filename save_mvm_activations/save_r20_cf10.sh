#!/bin/bash

python save_mvm_resnet20.py --mvm --nideal -b2000 --dataset='cifar10' --mode='train' --pretrained='../pretrained_models/ideal/resnet20fp_cifar10.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/rram/multiple_batches/' 

python save_mvm_resnet20.py --mvm --nideal -b2000 --dataset='cifar10' --mode='test' --pretrained='../pretrained_models/ideal/resnet20fp_cifar10.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/rram/multiple_batches/' 

python save_mvm_resnet20.py --mvm -b2000 --dataset='cifar10' --mode='train' --pretrained='../pretrained_models/ideal/resnet20fp_cifar10.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/sram/multiple_batches/' 

python save_mvm_resnet20.py --mvm -b2000 --dataset='cifar10' --mode='test' --pretrained='../pretrained_models/ideal/resnet20fp_cifar10.pth.tar' --savedir='/home/nano01/a/esoufler/activations/x64-8b/sram/multiple_batches/'