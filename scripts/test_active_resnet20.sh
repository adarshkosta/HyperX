#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar100.pth.tar"
GPUS="2,3"

for ((f=19; f>0; f=f-2))
do
    python ../error_analysis/test_active.py --dataset=$DATASET --pretrained=$PRETRAINED --active-layers=$f --gpus=$GPUS --mvm --nideal -b512 -j16
done

f=0
python ../error_analysis/test_active.py --dataset=$DATASET --pretrained=$PRETRAINED --active-layers=$f --gpus=$GPUS --mvm --nideal -b512 -j16