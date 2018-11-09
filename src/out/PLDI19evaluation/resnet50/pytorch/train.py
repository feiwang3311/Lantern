from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inputs
import resnet50
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.onnx

def train(args):
  startTime = time.time()
  cudnn.benchmark = True
  cudnn.deterministic = True
  torch.set_num_threads(1)
  torch.manual_seed(args.seed)

  model = resnet50.resnet50Cifar10()
  if args.use_gpu:
    model.cuda()
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  batch = inputs.Batch(args.input_file, args.batch_size)

  def train_epoch(epoch):
    tloss = 0.0
    for i in range(batch.total_size // batch.batch_size):
      (input_x, input_y) = batch.batch()
      inputX = Variable(torch.from_numpy(input_x))
      inputY = Variable(torch.from_numpy(input_y))
      if args.use_gpu:
        inputX = inputX.cuda()
        inputY = inputY.cuda()
      optimizer.zero_grad()
      loss = F.nll_loss(F.log_softmax(model(inputX), dim=1), inputY)
      tloss += loss.data.item()
      loss.backward()
      optimizer.step()
      if (i + 1) % (batch.total_size // batch.batch_size // 10) == 0:
        print('epoch %d: step %d, training loss %f' % (epoch + 1, i + 1, tloss / (i)))
    return tloss / (batch.batch_size)

  def inference_epoch(epoch):
    model.eval()
    for i in range(batch.total_size // batch.batch_size):
      (input_x, input_y) = batch.batch()
      inputX = Variable(torch.from_numpy(input_x))
      if args.use_gpu:
        inputX = inputX.cuda()
      resnet50.printHead(10, inputX, "input")
      res = model(inputX)
      resnet50.printHead(10, res, "output")
      exit(0)
      if (i + 1) % (batch.total_size // batch.batch_size // 10) == 0:
        print('epoch %d: step %d, training loss %f' % (epoch + 1, i + 1, tloss / (i)))
    return 0

  loopStart = time.time()
  loss_save = []
  for epoch in range(args.epochs):
    start = time.time()
    if args.inference:
      loss_save.append(inference_epoch(epoch))
    else:
      loss_save.append(train_epoch(epoch))
    stop = time.time()
    print('Training completed in {} sec ({} sec/image)'.format(int(stop - start), (stop - start)/60000))
  loopEnd = time.time()

  prepareTime = loopStart - startTime
  loopTime = loopEnd - loopStart
  timePerEpoch = loopTime / args.epochs

  with open(args.write_to, "w") as f:
    f.write("unit: " + "1 epoch\n")
    for loss in loss_save:
      f.write("{}\n".format(loss))
    f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")


if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
            help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
            help='learning rate (default: 0.005)')
  parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
            help='SGD momentum (default: 0.0)')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--input_file', type=str,
           default='../../cifar10_data/cifar-10-batches-py/data_batch_1',
           help='Directory for storing input data')
  parser.add_argument('--write_to', type=str,
           default='result_PyTorch',
           help='Directory for saving performance data')
  parser.add_argument('--generate_onnx', type=str, default='',
           help='Directory for saving ONNX model')
  parser.add_argument('--use_gpu', type=bool, default=False,
           help='Set to true if you want to use GPU')
  parser.add_argument('--inference', type=bool, default=False,
           help='Set to false if you want to measure inference time')
  args = parser.parse_args()

  if args.generate_onnx == '':
    train(args)
  else:
    torch.manual_seed(args.seed)
    model = resnet50.resnet50Cifar10()
    batch = inputs.Batch('../../cifar10_data/cifar-10-batches-py/data_batch_1', 64)
    (input_x, input_y) = batch.batch()
    torch.onnx.export(model, Variable(torch.from_numpy(input_x)), args.generate_onnx, verbose=True)
