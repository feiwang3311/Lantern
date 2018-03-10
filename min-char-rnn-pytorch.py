"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }
def char_to_ix(ch):
  return ord(ch) - ord('a')
def ix_to_char(ix):
  return chr(ix + ord('a'))

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 20 # number of steps to unroll the RNN for
learning_rate = 1e-5

# import relevant supports
import torch

def lineToTensor(line):
  tensor = torch.zeros(len(line), 1, vocab_size)
  for li, letter in enumerate(line):
    tensor[li][0][char_to_ix(letter)] = 1
  return tensor

def lineToLongTensor(line):
  tensor = torch.LongTensor(len(line), 1).zero_()
  for li, letter in enumerate(line):
    tensor[li][0] = char_to_ix(letter)
  return tensor

import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size

    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, input, hidden):
    #print(input)
    #print(hidden)
    combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(hidden)
    output = self.softmax(output)
    return output, hidden

  def initHidden(self):
    return Variable(torch.zeros(1, self.hidden_size))

rnn = RNN(vocab_size, hidden_size, vocab_size)      

criterion = nn.NLLLoss()

def train(output_tensor, input_tensor):
  hidden = rnn.initHidden()

  rnn.zero_grad()

  loss = 0

  for i in range(input_tensor.size()[0]):
      output, hidden = rnn(input_tensor[i], hidden)
      if (i == 0): 
        loss = criterion(output, output_tensor[i])
      else:
        loss = loss + criterion(output, output_tensor[i]) 

  loss.backward()

  # Add parameters' gradients to their values, multiplied by learning rate
  for p in rnn.parameters():
      p.data.add_(-learning_rate, p.grad.data)

  return output, loss.data[0]

p = 0
#mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
#mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)

for iter in range(1, 2000 + 1):

  if p+seq_length+1 >= len(data): p = 0

  inputs  = Variable(lineToTensor(data[p:p+seq_length]))
  targets = Variable(lineToLongTensor(data[p+1:p+seq_length+1]))
  output, loss = train(targets, inputs)

  # Print iter number, loss, name and guess
  if iter % 100 == 0: print('iter %d, loss: %f' % (iter, loss))

  p += seq_length

"""
n, p = 0, 0
mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)* seq_length
while True:
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = Variable(torch.zeros((hidden_size,1)), requires_grad=False)
    p = 0
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  loss, hprev = lossFunT(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))

  # perform parameter update with Adagrad
  for param, dparam, mem in zip ([Wxh, Whh, Why, bh, by],
                                 [dWxh, dWhh, dWhy, dbh, dby],
                                 [mWxh, mWhh, mWhy, mbh, mby]):
    mem.add_(dparam * dparam)
    param.add_(-learning_rate * dparam / torch.sqrt(mem + 1e-8))
    dparam.zero_() #

  p += seq_length
  n += 1
"""
