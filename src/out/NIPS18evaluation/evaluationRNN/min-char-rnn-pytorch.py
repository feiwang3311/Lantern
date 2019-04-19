"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def run(write_to):

  torch.set_num_threads(1)

  start = time.time()
  data = open('graham.txt', 'r').read() # should be simple plain text file
  chars = list(set(data))
  data_size, vocab_size = len(data), len(chars)
  print('data has %d characters, %d unique.' % (data_size, vocab_size))
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }

  # hyper-parameters
  hidden_size = 50 # size of hidden layer of neurons
  seq_length = 20 # number of steps to unroll the RNN for
  batch_size = 20
  learning_rate = 1e-1
  n_iter = 5000
  iter_step = 100

  torch.manual_seed(1)

  def lineToTensor(line):
    tensor = torch.zeros(seq_length, batch_size, vocab_size)
    for i in range(seq_length):
      for j in range(batch_size):
        tensor[i][j][char_to_ix[line[j * seq_length + i]]] = 1
    return tensor

  def lineToLongTensor(line):
    tensor = torch.LongTensor(seq_length, batch_size).zero_()
    for i in range(seq_length):
      for j in range(batch_size):
        tensor[i][j] = char_to_ix[line[j * seq_length + i]]
    return tensor

  class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(RNN, self).__init__()

      self.hidden_size = hidden_size
      self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
      self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
      combined = torch.cat((input, hidden), 1)
      hidden = torch.tanh(self.i2h(combined))
      output = self.i2o(hidden)
      return output, hidden

    def initHidden(self):
      return Variable(torch.zeros(batch_size, self.hidden_size))

  rnn = RNN(vocab_size, hidden_size, vocab_size)
  optimizer = torch.optim.Adagrad(rnn.parameters(), lr = learning_rate)
  criterion = nn.CrossEntropyLoss()

  def train(output_tensor, input_tensor):
    hidden = rnn.initHidden()

    optimizer.zero_grad()

    loss = 0

    for i in range(input_tensor.size()[0]):
      output, hidden = rnn(input_tensor[i], hidden)
      loss += criterion(output, output_tensor[i])

    loss.backward()

    # grad clipping and stepping
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0, norm_type=1)
    optimizer.step()

    return loss.item()

  end = time.time()
  prepareTime = end-start

  loss_save = []
  p = -seq_length * batch_size
  start = time.time()
  for iter in range(n_iter + 1):
    p += seq_length * batch_size
    if p+seq_length * batch_size+1 >= len(data): p = 0

    inputs  = Variable(lineToTensor(data[p:p+seq_length * batch_size]))
    targets = Variable(lineToLongTensor(data[p+1:p+seq_length * batch_size +1]))
    loss = train(targets, inputs)
    if iter % iter_step == 0:
      print('iter %d, loss: %f' % (iter, loss))
      loss_save.append(loss)

  end = time.time()
  loopTime = end -start

  with open(write_to, "w") as f:
    f.write("unit: " + "100 iteration\n")
    for loss in loss_save:
      f.write("{}\n".format(loss))
    f.write("run time: " + str(prepareTime) + " " + str(loopTime) + "\n")

if __name__ == '__main__':
  import sys
  if (len(sys.argv) != 2):
    print("should have a file to write results to")
    exit(0)
  run(sys.argv[1])
