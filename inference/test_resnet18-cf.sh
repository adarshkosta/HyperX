#!/bin/bash

DATASET="cifar100"
PRETRAINED="../pretrained_models/frozen/x64-8b/"
LOADDIR="/home/nano01/a/esoufler/activations/x64-8b/"
MODE="$1"
GPUS="$2"
START=$3
STOP=$4

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../inference/test_resnet18-cf.py --mvm --dataset=$DATASET --mode=$MODE --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR
done