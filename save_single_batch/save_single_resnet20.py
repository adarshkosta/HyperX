import argparse
import torch
import numpy as np
import os
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--b-train', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1000)')
parser.add_argument('--b-test', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1000)')
parser.add_argument('--datadir', default='/home/nano01/a/esoufler/activations/x128-8b/rram/', help='dataset name or folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', help='name of the model')
parser.add_argument('--mode', metavar='MODE', default='both', help='set to save')
args = parser.parse_args()

train_length = 50000
test_length = 10000
#%% 
if args.mode != 'test':
#    TRAIN LABELS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/labels/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/labels/')

    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_train
    num = int(train_length/batch_size)
    
    print('Saving Train labels...')
    for batch in trange(0, num):
    #    print(batch)
        filename = os.path.join(base_src_path, 'labels_' + str(batch) + '.pth.tar')
        src_value = torch.load(filename)
    
        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1
            
    # TRAIN ACTIVATIONS 
    print('Saving Train activations...')
    for i in range(1,20,2):
        # trainset
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/relu'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/relu'+ str(i) +'/')
        
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_train
        num = int(train_length/batch_size)
        
        print('relu'+str(i))
        for batch in trange(0, num):
            filename = os.path.join(base_src_path, 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1
    
    # FC ACTIVATIONS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/fc'+ '/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/fc'+ '/')
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
        
    batch_size = args.b_train
    num = int(train_length/batch_size)
    
    print('fc')
    for batch in trange(0, num):
        filename = os.path.join(base_src_path, 'act_fc' + '_' + str(batch) + '.pth.tar')
        src_value = torch.load(filename)
    
        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar')
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1

#%% 
if args.mode !='train':
    #TEST LABELS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/labels/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/labels/')
    
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
        
    batch_size = args.b_test
    num = int(test_length/batch_size)
    
    print('Saving Test labels...')
    for batch in trange(0, num):
    #    print(batch)
        filename = os.path.join(base_src_path, 'labels_' + str(batch) + '.pth.tar')
        src_value = torch.load(filename)
    
        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar') 
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1
            
    # TEST ACTIVATIONS
    print('Saving Test activations...')
    for i in range(1,20,2):
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/relu'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/relu'+ str(i) +'/')
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_test
        num = int(test_length/batch_size)
        
        print('relu'+str(i))
        for batch in trange(0, num):
    #        print(batch)
            filename = os.path.join(base_src_path, 'act_relu'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_relu'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1
    
    # FC ACTIVATIONS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/fc'+'/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/fc'+'/')
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
        
    batch_size = args.b_test
    num = int(test_length/batch_size)
    
    print('fc')
    for batch in trange(0, num):
    #        print(batch)
        filename = os.path.join(base_src_path, 'act_fc' +'_' + str(batch) + '.pth.tar') 
        src_value = torch.load(filename)
    
        element_in_index = 0
        for element in src_value.detach().cpu().numpy().tolist():
            dst_tensor = torch.Tensor([element])
            dst_filename = os.path.join(base_dst_path, 'act_fc' + '_' + str(batch_size * batch + element_in_index) + '.pth.tar')
            torch.save(dst_tensor, dst_filename)
            element_in_index += 1