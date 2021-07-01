# Hybrid_RRAM-SRAM

## Installation
Clone this repository using: 
```git clone https://github.com/adarshkosta/Hybrid_RRAM-SRAM.git```

The folders ```Hybrid_RRAM-SRAM/xb_models``` containing the crossbar models and ```Hybrid_RRAM-SRAM/pretrained_models``` need to be present and can be downloaded from my Onedrive.

## Inference
Codes for inference (ideal, ideal_mvm, non-ideal_mvm) are in the inference folder.
For example: To run inference for ResNet-20 on CIFAR-10 with non-ideal_mvm use the following command.

```python test_resnet20.py --pretrained= <PATH TO MODEL> --mvm --nideal --dataset='cifar10'```

Other command-line arguments can be looked upon from the code.

## Saving activations
For saving non-ideal mvm activations for ResNet-20 on CIFAR-10 for the training set, use the following command.
```python save_mvm_resnet20.py --pretrained='path to saved model' --dataset='cifar10' --mode='test' --gpus='0,1,2,3' -b1000 -j16 --exp='resnet20_cf10_test' --nideal --mvm```

##Quantization
###Dorefa-Training
From the training folder, run:
```python train_q_resnet20.p```
Check for proper command line arguments.

###Dorefa-models for Resnet20
```pretrained_models/ideal/resnet20qfp_cifar10_i8b4f_w8b6f.pth.tar```

```pretrained_models/ideal/resnet20qfp_cifar10_i8b5f_w8b7f.pth.tar```

###Testing a dorefa trained network on func-sim (ideal + non-ideal mode)
From the inference folder, run:
```python test_quant.py --quantize-model --mvm```

```python test_quant.py --quantize-model --mvm --nideal```

###Save-quantized-non-ideal activations
From save_mvm_activations, run:
```python save_mvm_resnet20.py --mvm --nideal --quantize-model --pretrained=$PATH_TO_MODEL --savedir=$SAVEDIR``` 

###Save-single-batch
From save_single_batch folder, run:
```python save_single_resnet20.py --dataset=$DATASET```

###Retraining SRAM-layers
From retraining folder, run:
```python retrain_resnet20_quantized.py --frozen-layers=$F --pretrained=$PATH_TO_MODEL --loaddir=$LOADDIR --savedir=$SAVEDIR```

###Rememeber to update ```config.py``` with appropriate weight and input bits and fractions as well as corresponding crossbar configurations when using func-sim (not needed when retraining SRAM layers).

NOTE: Please use all 4 GPUs on the server (0,1,2,3) and batch_size=1000, else code wouldn't work. Currently only fixed for 4 GPUs operating together.

## Some useful Command-line arguments 

```--dataset``` : dataset (CIFAR-10/100)

```--savedir``` : base-directory for saving activations

```--workers, -j``` : number of workers to use

```--gpus``` : GPU IDs to use

```--mode``` : save 'train' or 'test' set activations

```--pretrained``` : path to pretrained model

```--batch-size, -b``` : batch-size

```--mvm``` : use functional-simulator backend

```--nideal``` : add crossbar non-idealities

Further additions coming soon!
