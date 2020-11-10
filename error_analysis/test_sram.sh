#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar100.pth.tar"
TYPE="$1"
START=$2
GPUS="$3"

for ((f=$START; f<20; f=f+2))
do
    python test_sram.py --dataset=$DATASET --pretrained=$PRETRAINED --type=$TYPE --frozen-layers=$f --gpus=$GPUS -b1024 -j16
done

f=20

python test_sram.py --dataset=$DATASET --pretrained=$PRETRAINED --type=$TYPE --frozen-layers=$f --gpus=$GPUS  -b1024 -j16
