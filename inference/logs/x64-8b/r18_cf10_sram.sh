WARNING: crossbar sizes with different row annd column dimension not supported.

      ==> Arguments:
          dataset: cifar10
          model: resnet18
          load_dir: /home/nano01/a/esoufler/activations/x64-8b/
          mode: sram
          savedir: ../results/discp_analysis/
          pretrained: ../pretrained_models/frozen/x64-8b/
          mvm: True
          nideal: None
          workers: 8
          batch_size: 64
          print_freq: 5
          half: None
          gpus: 3
          frozen_layers: 1

      ==> Functional simulator configurations:
          weight_bits=8
          weight_bit_frac=6
          input_bits=8
          input_bit_frac=6
          xbar_row_size=64
          xbar_col_size=64
          tile_row=2
          tile_col=2
          bit_stream=1
          bit_slice=2
          adc_bit=14
          acm_bits=32
          acm_bit_frac=24
          mvm=True
          non-ideality=False
          
xbmodel=NN_model(
  (fc1): Linear(in_features=4160, out_features=500, bias=True)
  (relu1): ReLU(inplace=True)
  (do2): Dropout(p=0.5, inplace=False)
  (fc3): Linear(in_features=500, out_features=64, bias=True)
)
          
xbmodel_weight_path=../xb_models/XB_64_stream1slice207dropout50epochs.pth.tar
          inmax_test=1.2
          inmin_test=0.857


DEVICE: cuda
GPU Id(s) being used: 3
==> Building model for resnet18 ...
WARNING: crossbar sizes with different row annd column dimension not supported.
==> Initializing model parameters ...
==> Load pretrained model form ../pretrained_models/frozen/x64-8b/rram/cifar10/resnet18/freeze1_hp_best.pth.tar ...
Pretrained model accuracy: 95.02999877929688
