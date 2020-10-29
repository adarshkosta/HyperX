import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-b_train', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1024)')
parser.add_argument('-b_test', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1024)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', help='name of the model')
args = parser.parse_args()

#%%
################ SAVE THE LABELS (we do it only once)
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)

batch_size = args.b_train
num = int(50000/batch_size)

print('Saving Train labels...')
for batch in range(num):
#    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

#%%
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

print('Saving Test labels...')
for batch in range(num):
#    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

#%% 
################ SAVE THE ACTIVATIONS 
print('Saving Training and Test activations...')
for i in range(1,20,2):
    # trainset
    base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/relu'+ str(i) +'/'
    base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/relu'+ str(i) +'/'
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_train
    num = int(50000/batch_size)
    
    print('relu'+str(i))
    for batch in range(num):
        filename = base_src_path + 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar' 
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = base_dst_path + 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

    # testset
    base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/relu'+ str(i) +'/'
    base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/relu'+ str(i) +'/'
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_test
    num = int(10000/batch_size)

    for batch in range(num):
#        print(batch)
        filename = base_src_path + 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar' 
        src_value = torch.load(filename)

        element_in_index = 0
        for element in src_value.cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = base_dst_path + 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

#%%
#################SAVE FC ACTIVATINS-----------------------------
# trainset
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/fc'+ '/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/fc'+ '/'

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_train
num = int(50000/batch_size)

print('fc')
for batch in range(num):
    filename = base_src_path + 'act_fc' + '_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1

# testset
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/fc'+'/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/fc'+'/'

if not os.path.exists(base_dst_path):
    os.makedirs(base_dst_path)
    
batch_size = args.b_test
num = int(10000/batch_size)

for batch in range(num):
#        print(batch)
    filename = base_src_path + 'act_fc' +'_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1
