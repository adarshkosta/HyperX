import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--b-train', default=64, type=int,
                     metavar='N', help='mini-batch size (default: 1024)')
parser.add_argument('--b-test', default=64, type=int,
                     metavar='N', help='mini-batch size (default: 1024)')
parser.add_argument('--datadir', default='/home/nano01/a/esoufler/activations/', help='dataset name or folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='mobilenet', help='name of the model')
parser.add_argument('--mode', metavar='MODE', default='both', help='set to save')
args = parser.parse_args()

print(args)

#%% 
if args.mode != 'test':
#    TRAIN LABELS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/labels/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/labels/')
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
    
    batch_size = args.b_train
    num = int(50000/batch_size)
    
    print('Saving Train labels...')
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
            
    # TRAIN ACTIVATIONS 
    print('Saving Train activations...')
    for i in range(1,18):
        # trainset
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/InvertedResidual'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/InvertedResidual'+ str(i) +'/')
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_train
        num = int(50000/batch_size)
        
        print('InvertedResidual'+str(i))
        for batch in range(num):
            filename = os.path.join(base_src_path, 'act_InvertedResidual'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_InvertedResidual'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1

    for i in range(1,3):
        # trainset
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/ConvBNReLU'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/ConvBNReLU'+ str(i) +'/')
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_train
        num = int(50000/batch_size)
        
        print('ConvBNReLU'+str(i))
        for batch in range(num):
            filename = os.path.join(base_src_path, 'act_ConvBNReLU'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_ConvBNReLU'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1
    
    # FC ACTIVATIONS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'train/fc'+ '/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'train/fc'+ '/')
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
        
    batch_size = args.b_train
    num = int(50000/batch_size)
    
    print('fc')
    for batch in range(num):
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
            
    # TEST ACTIVATIONS
    print('Saving Test activations...')
    for i in range(1,18):
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/InvertedResidual'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/InvertedResidual'+ str(i) +'/')
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_test
        num = int(10000/batch_size)
        
        print('InvertedResidual'+str(i))
        for batch in range(num):
    #        print(batch)
            filename = os.path.join(base_src_path, 'act_InvertedResidual'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_InvertedResidual'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1

    for i in range(1,3):
        # trainset
        base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/ConvBNReLU'+ str(i) +'/')
        base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/ConvBNReLU'+ str(i) +'/')
        
        if not os.path.exists(base_dst_path):
            os.makedirs(base_dst_path)
        
        batch_size = args.b_test
        num = int(10000/batch_size)
        
        print('ConvBNReLU'+str(i))
        for batch in range(num):
            filename = os.path.join(base_src_path, 'act_ConvBNReLU'+ str(i) +'_' + str(batch) + '.pth.tar')
            src_value = torch.load(filename)
    
            element_in_index = 0
            for element in src_value.detach().cpu().numpy().tolist():
                dst_tensor = torch.Tensor([element])
                dst_filename = os.path.join(base_dst_path, 'act_ConvBNReLU'+ str(i) +'_' + str(batch_size * batch + element_in_index) + '.pth.tar')
                torch.save(dst_tensor, dst_filename)
                element_in_index += 1
    
    # FC ACTIVATIONS
    base_src_path = os.path.join(args.datadir, 'multiple_batches', str(args.dataset), str(args.model), 'test/fc'+'/')
    base_dst_path = os.path.join(args.datadir, 'one_batch', str(args.dataset), str(args.model), 'test/fc'+'/')
    
    if not os.path.exists(base_dst_path):
        os.makedirs(base_dst_path)
        
    batch_size = args.b_test
    num = int(10000/batch_size)
    
    print('fc')
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