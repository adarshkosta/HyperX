#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet20fp_cifar100.pth.tar"
LOADDIR="/home/nano01/a/esoufler/activations/x64/rram/one_batch/"
SAVEDIR="../pretrained_models/frozen/x64/"
GPUS="$1"
START=$2
STOP=$3
LR=$4

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../retraining/retrain_resnet20.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR --savedir=$SAVEDIR --lr=$LR
done