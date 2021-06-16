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
