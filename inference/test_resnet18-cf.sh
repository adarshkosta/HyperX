#!/bin/bash

DATASET="cifar10"
PRETRAINED="../pretrained_models/frozen/x64-8b/"
LOADDIR="/home/nano01/a/esoufler/activations/x64-8b/"
MODE="rram"
GPUS="$1"
START=$2
STOP=$3

for ((f=$START; f<=$STOP; f=f+2))
do
    python ../inference/test_resnet18-cf.py --mvm --dataset=$DATASET --mode=$MODE --pretrained=$PRETRAINED --frozen-layers=$f --gpus=$GPUS --load-dir=$LOADDIR
done
