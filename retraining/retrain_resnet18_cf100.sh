#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/resnet18fp_imnet.pth.tar"
LOADDIR="/home/nano01/a/esoufler/activations/x64-8b/"
SAVEDIR="../pretrained_models/frozen/x64-8b/"
MODE="$1"
GPUS="$2"
START=$3
STOP=$4
LR=$5

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../retraining/retrain_resnet18_quantized.py --mode=$MODE --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR --savedir=$SAVEDIR --lr=$LR
done
