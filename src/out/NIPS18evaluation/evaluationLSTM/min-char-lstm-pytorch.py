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
  # data I/O
  torch.set_num_threads(1)

  start = time.time()
  data = open('graham.txt', 'r').read() # should be simple plain text file
  chars = list(set(data))
  data_size, vocab_size = len(data), len(chars)
  print('data has %d characters, %d unique.' % (data_size, vocab_size))
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }

  # hyperparameters
  hidden_size = 50 # size of hidden layer of neurons
  seq_length = 20 # number of steps to unroll the RNN for
  learning_rate = 1e-1
  n_iters = 5000
  iter_step = 100
  batch_size = 20

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

  def lineToLongTensor1D(line):
    tensor = torch.LongTensor(seq_length, batch_size).zero_()
    for i in range(seq_length):
      for j in range(batch_size):
        tensor[i][j] = char_to_ix[line[j * seq_length + i]]
    return tensor.view(-1)

    # tensor = torch.LongTensor(len(line)).zero_()
    # for li, letter in enumerate(line):
    #   tensor[li] = char_to_ix[letter]
    # return tensor

  class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(RNN, self).__init__()

      self.hidden_size = hidden_size
      self.lstm = nn.LSTM(input_size, hidden_size)
      self.i2o = nn.Linear(hidden_size, output_size)
      self.softmax = nn.LogSoftmax(dim=1)
      self.hidden = self.init_hidden()

    def forward(self, inputs):# inputs is 1D chars
      inputsv = Variable(lineToTensor(inputs)) # inputsv is 3D
      lstm_out, self.hidden = self.lstm(inputsv, self.hidden)
      tag_space = self.i2o(lstm_out.view(-1, hidden_size))
      tag_scores = F.log_softmax(tag_space, dim=1)
      return tag_scores

    def init_hidden(self):
      return (Variable(torch.zeros(1, batch_size, self.hidden_size)),
              Variable(torch.zeros(1, batch_size, self.hidden_size)))

  model = RNN(vocab_size, hidden_size, vocab_size)
  loss_function = nn.NLLLoss(size_average=False, reduce=True)
  optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

  end = time.time()
  prepareTime = end - start

  p = -seq_length * batch_size

  start = time.time()
  loss_save = []
  for iter in range(n_iters + 1):
    p += seq_length * batch_size
    if p+seq_length * batch_size +1 >= len(data) or (iter == 0):
      p = 0

    inputs  = data[p:p+seq_length * batch_size]
    targets = data[p+1:p+seq_length * batch_size+1]

    model.zero_grad()
    model.hidden = model.init_hidden()
    tag_scores = model(inputs)
    loss = loss_function(tag_scores, Variable(lineToLongTensor1D(targets)))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 5.0, norm_type=1)
    optimizer.step()

    loss = loss.data[0]
    if iter % iter_step == 0:
      print('iter %d, loss: %f' % (iter, loss))
      loss_save.append(loss)

  end = time.time()
  loopTime = end - start

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
