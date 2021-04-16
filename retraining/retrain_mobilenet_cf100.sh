#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/ideal/mobilenetv2fp_imnet.pth.tar"
LOADDIR="/home/nano01/a/esoufler/activations/sram/one_batch/"
GPUS="$1"
START=$2
STOP=$3

for ((f=$START; f<$STOP+1; f=f+1))
do
    python ../retraining/retrain_mobilenet.py --dataset=$DATASET --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR
done
