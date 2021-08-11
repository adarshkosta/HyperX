#!/bin/bash

DATASET="cifar10"
PRETRAINED="../pretrained_models/ideal/resnet20qfp_cifar10_i8b4f_w8b6f.pth.tar"
LOADDIR="/home/nano01/a/esoufler/activations/"
SAVEDIR="../pretrained_models/frozen/"
EXP="x128-8b"
MODE="$1"
GPUS="$2"
START=$3
STOP=$4
LR=$5

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../retraining/retrain_resnet20_quantized.py --exp=$EXP --mode=$MODE --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR --savedir=$SAVEDIR --lr=$LR
done