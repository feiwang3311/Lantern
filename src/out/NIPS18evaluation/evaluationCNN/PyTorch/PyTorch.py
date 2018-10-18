from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

def run(write_to):

  startTime = time.time()

  torch.set_num_threads(1)
  torch.manual_seed(args.seed)
  kwargs = {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
             transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
             # transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

  print(len(train_loader))  # should be 600

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
      self.fc1 = nn.Linear(320, 50)
      self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = x.view(-1, 320)
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = self.fc2(x)
      return F.log_softmax(x, dim=1)

    def print(self):
      print("== conv1 ==")
      print('min {} - max {}'.format(self.conv1.weight.min().data[0], self.conv1.weight.max().data[0]))
      print("== conv2 ==")
      print('min {} - max {}'.format(self.conv2.weight.min().data[0], self.conv2.weight.max().data[0]))
      print("== fc1 ==")
      print('min {} - max {}'.format(self.fc1.weight.min().data[0], self.fc1.weight.max().data[0]))
      print('min {} - max {}'.format(self.fc1.bias.min().data[0], self.fc1.bias.max().data[0]))
      print("== fc2 ==")
      print('min {} - max {}'.format(self.fc2.weight.min().data[0], self.fc2.weight.max().data[0]))
      print('min {} - max {}'.format(self.fc2.bias.min().data[0], self.fc2.bias.max().data[0]))

  model = Net()
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  def train(epoch):
    # model.train()
    tloss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = Variable(data), Variable(target)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      tloss += loss.data[0]
      loss.backward()
      optimizer.step()
      if (batch_idx + 1) % (args.log_interval / args.batch_size) == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), tloss / (batch_idx)))
    return tloss / (batch_idx)

  loopStart = time.time()
  loss_save = []
  for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss_save.append(train(epoch))
    stop = time.time()
    print('Training completed in {} sec ({} sec/image)'.format(int(stop - start), (stop - start)/60000))
  loopEnd = time.time()

  prepareTime = loopStart - startTime
  loopTime = loopEnd - loopStart
  timePerEpoch = loopTime / args.epochs

  with open(write_to, "w") as f:
    f.write("unit: " + "1 epoch\n")
    for loss in loss_save:
      f.write("{}\n".format(loss))
    f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")


if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=100, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
            help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
            help='learning rate (default: 0.05)')
  parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
            help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
            help='random seed (default: 1)')
  ## Note NEED default to be the same as total data length, or 1/10 of the total data length
  parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
            help='how many batches to wait before logging training status')
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  run("result_PyTorch"+str(args.batch_size)+".txt")