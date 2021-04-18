#!/bin/bash

DATASET="cifar10"
PRETRAINED="../pretrained_models/frozen/x128/"
LOADDIR="/home/nano01/a/esoufler/activations/x128/"
MODE="rram_new"
GPUS="$1"
START=$2
STOP=$3

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../inference/test_resnet18-cf.py --dataset=$DATASET --mode=$MODE --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR
done
