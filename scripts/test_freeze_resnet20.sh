#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar100.pth.tar"
GPUS="0,1"
START=$1

for ((f=1; f<20; f=f+2))
do
    python ../error_analysis/test_freeze.py --dataset=$DATASET --pretrained=$PRETRAINED --rram-layers=$f --gpus=$GPUS --mvm --nideal -b512 -j16
done

f=20

python ../error_analysis/test_freeze.py --dataset=$DATASET --pretrained=$PRETRAINED --rram-layers=$f --gpus=$GPUS --mvm --nideal -b512 -j16
