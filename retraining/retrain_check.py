import os
import sys
import time
import collections

#Filepath handling
root_dir = os.path.dirname(os.getcwd())
inference_dir = os.path.join(root_dir, "inference")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, inference_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

#%%
# Standard or Built-in packages
import numpy as np
import random
import argparse
import pdb

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

#torch.set_default_tensor_type(torch.HalfTensor)

# User-defined packages
import models
from utils.data import get_dataset
from utils.preprocess import get_transform
from utils.utils import *
import src.config as cfg
from utils_bar import progress_bar

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *

#Seeding
new_manual_seed = 0
torch.manual_seed(new_manual_seed)
torch.cuda.manual_seed_all(new_manual_seed)
np.random.seed(new_manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(new_manual_seed)
os.environ['PYTHONHASHSEED'] = str(new_manual_seed)

#Create available models list
model_names = []
for path, dirs, files in os.walk(models_dir):
    for file_n in files:
        if (not file_n.startswith("__")):
            model_names.append(file_n.split('.')[0])
    break # only traverse top level directory
model_names.sort()


# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(trainloader):
        inputs = batch['data']
        targets = batch['target'].type(torch.LongTensor)

        if args.precision:
                inputs = inputs.half()
                targets = targets.half()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Evaluate
def test(net, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            inputs = batch['data']
            targets = batch['target'].type(torch.LongTensor)

            if args.precision:
                inputs = inputs.half()
                targets = targets.half()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+str(args.dataset)+'_'+str(args.model)+'_freeze_'+str(args.frozen_layers)+'.pth')
        best_acc = acc
    print('Best Inference Accuracy : ', best_acc)
  


class MyDataParallel(nn.DataParallel):
    def __getattr(self, name):
        return getattr(self.module, name)
    
class SplitActivations_Dataset(Dataset):
    def __init__(self, datapath, tgtpath, train_len = True):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        
    def __getitem__(self, index):
        x = torch.load(self.datapath+'/act_relu'+str(args.frozen_layers)+'_'+str(index)+'.pth.tar')
        y = torch.load(self.tgtpath+'/labels_'+str(index)+'.pth.tar')
        return {'data': x[0], 'target': y[0]}
    
    def __len__(self):
        if self.train_len == True:
            return 50000 
        else :
            return 10000


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                help='mini-batch size (default: 256)')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                help='dataset name or folder')
    parser.add_argument('--loaddir', default='/home/nano01/a/esoufler/activations/one_batch/',
                help='base path for loading activations')
    parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20',
                choices=model_names,
                help='name of the model')
    parser.add_argument('--pretrained', action='store', default='../pretrained_models/ideal/resnet20fp_cifar100.pth.tar',
                help='the path to the pretrained model')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='J',
                help='number of data loading workers (default: 8)')
    parser.add_argument('--gpus', default='0,1,2,3', help='gpus (default: 4)')
    parser.add_argument('-frozen_layers', default=1, type=int, help='number of frozen layers in the model')
    parser.add_argument('-epochs', default=180, type=int, help='number of epochs to train the model')
    parser.add_argument('-precision', default=False, help='floating point 32 (False) or floating point 16 (True)')
    parser.add_argument('-load_pretrained_model', default=False, help='fload pretrained weights or random initialization')
    args = parser.parse_args()
    
    print('\n' + ' '*6 + '==> Arguments:')
    for k, v in vars(args).items():
        print(' '*10 + k + ': ' + str(v))
    
    
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    print('GPU Id(s) being used:', args.gpus)

    print('==> Building model for', args.model, '...')
    if (args.model in model_names):
        model = (__import__(args.model+'_freeze'+str(args.frozen_layers))) #import module using the string/variable_name
    else:
        raise Exception(args.model+'is currently not supported')
        
    if args.dataset == 'cifar10':
        model = model.net(num_classes=10)
    elif args.dataset == 'cifar100':
        model = model.net(num_classes=100)
    else:
      raise Exception(args.dataset + 'is currently not supported')

    
    if (args.load_pretrained_model == 'True'):
        print('==> Initializing model with pre-trained parameters ...')
        weights_conv = []
        weights_lin = []
        bn_data = []
        bn_bias = []
        running_mean = []
        running_var = []
        num_batches = []
        original_model = (__import__(args.model))
        if args.dataset == 'cifar10':
            original_model = original_model.net(num_classes=10)
        elif args.dataset == 'cifar100':
            original_model = original_model.net(num_classes=100)

        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        original_model.load_state_dict(pretrained_model['state_dict'])
        temp = False
        counter = 0
        counter_conv = 0
        counter_bn = 0
        counter_linear = 0


        counter_conv = 0
        for m in original_model.modules():
            if isinstance(m, nn.Conv2d):
                if (counter_conv >= args.frozen_layers):
                    temp = True
                    weights_conv.append(m.weight.data.clone())     
                counter_conv += 1          
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if (temp == True):
                    bn_data.append(m.weight.data.clone())
                    bn_bias.append(m.bias.data.clone())
                    running_mean.append(m.running_mean.data.clone())
                    running_var.append(m.running_var.data.clone())
                    num_batches.append(m.num_batches_tracked.clone())
            elif isinstance(m, nn.Linear):
                if (temp == True):
                    weights_lin.append(m.weight.data.clone())

        i=j=k=0
        for m in model.modules():
            if isinstance(m, (nn.Conv2d)):
                m.weight.data = weights_conv[i]
                i = i+1
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data = bn_data[j]
                m.bias.data = bn_bias[j]
                m.running_mean.data = running_mean[j]
                m.running_var.data = running_var[j]
                m.num_batches_tracked = num_batches[j]
                j = j+1
            elif isinstance(m, nn.Linear):
                m.weight.data = weights_lin[k]
                k=k+1


    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.data.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                   m.bias.data.uniform_(-stdv, stdv)



    if args.precision:
        model.to(device).half()
    else:
        model.to(device)

    model = nn.DataParallel(model)


    datapath_train = str(args.loaddir)+str(args.dataset)+'/'+str(args.model)+'/train/relu' + str(args.frozen_layers)
    tgtpath_train = str(args.loaddir)+str(args.dataset)+'/'+str(args.model)+'/train/labels' 

    datapath_test = str(args.loaddir)+str(args.dataset)+'/'+str(args.model)+'/test/relu' + str(args.frozen_layers)
    tgtpath_test = str(args.loaddir)+str(args.dataset)+'/'+str(args.model)+'/test/labels'


    train_data = SplitActivations_Dataset(datapath_train, tgtpath_train, train_len = True)
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    test_data = SplitActivations_Dataset(datapath_test, tgtpath_test, train_len = False)
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # if args.precision:
    #     optimizer = optimizer.half()

    best_acc = 0
    
    for epoch in range(0, args.epochs):
        train(model, epoch)
        test(model, epoch)
        if (epoch == 80):
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif (epoch == 120):
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


