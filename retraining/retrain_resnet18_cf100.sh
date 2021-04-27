#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet18fp_imnet.pth.tar"
LOADDIR="/home/nano01/a/esoufler/activations/x64-8b/"
SAVEDIR="../pretrained_models/frozen/x64-8b/"
GPUS="$1"
START=$2
STOP=$3
LR=$4

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../retraining/retrain_resnet18.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR --savedir=$SAVEDIR --lr=$LR
done
