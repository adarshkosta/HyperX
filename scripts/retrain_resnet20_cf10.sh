#!/bin/bash

DATASET="cifar10"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar10.pth.tar"
GPUS="$1"

for ((f=1; f<20; f=f+2))
do
    python ../retraining/retrain_resnet20.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS
done


