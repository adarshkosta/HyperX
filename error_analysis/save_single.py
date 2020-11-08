import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-b_test', default=250, type=int,
                     metavar='N', help='mini-batch size (default: 250)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', help='name of the model')
parser.add_argument('--type', '-t', metavar='TYPE', default='rram', help='name of the type')
args = parser.parse_args()

#%%
################ SAVE THE LABELS (we do it only once)
base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'labels')
base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'labels')

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

print('Saving Test labels...')
for batch in range(num):
#    print(batch)
    filename = os.path.join(base_src_path, 'labels_' + str(batch) + '.pth.tar')
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.detach().cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = os.path.join(base_dst_path, 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

#%% 
################ SAVE THE ACTIVATIONS 
print('Saving Test activations...')
for i in range(1,20,2):
    # testset
    base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'relu'+ str(i))
    base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'relu'+ str(i))
    
    print('relu' + str(i))

    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_test
    num = int(10000/batch_size)

    for batch in range(num):
#        print(batch)
        filename = os.path.join(base_src_path, 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar')
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

#%%
#################SAVE FC ACTIVATINS-----------------------------
# testset
base_src_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/multiple_batches/', args.dataset, args.type, 'fc')
base_dst_path = os.path.join('/home/nano01/a/esoufler/activations/error_analysis/one_batch/', args.dataset, args.type, 'fc')

print('fc')

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

for batch in range(num):
#        print(batch)
    filename = os.path.join(base_src_path, 'act_fc' +'_' + str(batch) + '.pth.tar')
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.detach().cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = os.path.join(base_dst_path, 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1
