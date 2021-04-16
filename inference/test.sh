#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar100.pth.tar"
GPUS="$1"

for ((f=1; f<20; f=f+2))
do
    python test.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS -b1000 -j16
done

f=20

python test.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS  -b1000 -j16
