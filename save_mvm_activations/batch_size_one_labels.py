import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-b_train', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1000)')
parser.add_argument('-b_test', default=1000, type=int,
                     metavar='N', help='mini-batch size (default: 1000)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20', help='name of the model')
args = parser.parse_args()

trainsize = 50000
testsize = 10000

################ SAVE THE LABELS (we do it only once)
base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'train/labels/'
batch_size = args.b_train

num = int(trainsize/batch_size)+1

for batch in range(num):
    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1
        

base_src_path = '/home/nano01/a/esoufler/activations/multiple_batches/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'
base_dst_path = '/home/nano01/a/esoufler/activations/one_batch/'+str(args.dataset)+'/'+str(args.model)+'/'+'test/labels/'
batch_size = args.b_test
num = int(testsize/batch_size)+1

for batch in range(num):
    print(batch)
    filename = base_src_path + 'labels_' + str(batch) + '.pth.tar' 
    src_value = torch.load(filename)

    element_in_index = 0
    for element in src_value.cpu().numpy().tolist():
        dst_tensor = torch.Tensor([element])
        dst_filename = base_dst_path + 'labels_' + str(batch_size * batch + element_in_index) + '.pth.tar' 
        torch.save(dst_tensor, dst_filename)
        element_in_index += 1