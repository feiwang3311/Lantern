from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import struct

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
          help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
          help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
          help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
          help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
          help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
          help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
          help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
          help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=True, download=True,
           transform= #transforms.Compose([
             transforms.ToTensor()# ,
             #transforms.Normalize((0.1307,), (0.3081,))
           ),#])),
  batch_size=1, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('../data', train=False, transform=
             transforms.ToTensor()),
  batch_size=1, shuffle=False, **kwargs)

import os
target_dir = '../data/bin/'
if not os.path.exists(target_dir):
  os.makedirs(target_dir)

def train():
  with open(target_dir + 'mnist_train.bin', 'wb') as f:
    with open(target_dir + 'mnist_train_target.bin', 'wb') as g:
      for batch_idx, (data, target) in enumerate(train_loader):
        for by in data.storage().tolist():
          f.write(struct.pack("@f", by))
        for by in target.storage().tolist():
          g.write(struct.pack("@i", int(by)))
        if batch_idx % args.log_interval == 0:
          print('[{}/{} ({:.0f}%)]'.format(
            batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader)))

def test():
  with open(target_dir + 'mnist_test.bin', 'wb') as f:
    with open(target_dir + 'mnist_test_target.bin', 'wb') as g:
      for data, target in test_loader:
        for by in data.storage().tolist():
          f.write(struct.pack("@f", by))
        for by in target.storage().tolist():
          g.write(struct.pack("@i", int(by)))

train()
test()