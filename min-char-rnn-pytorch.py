"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('graham.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
"""
def char_to_ix(ch):
  return ord(ch) - ord('a')
def ix_to_char(ix):
  return chr(ix + ord('a'))
"""

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 20 # number of steps to unroll the RNN for
learning_rate = 1e-1
n_epoch = 5000
epoch_step = 250

# import relevant supports
import torch
import torch.nn as nn
from torch.autograd import Variable


torch.manual_seed(1)

def lineToTensor(line):
  tensor = torch.zeros(len(line), 1, vocab_size)
  for li, letter in enumerate(line):
    tensor[li][0][char_to_ix[letter]] = 1
  return tensor

def lineToLongTensor(line):
  tensor = torch.LongTensor(len(line), 1).zero_()
  for li, letter in enumerate(line):
    tensor[li][0] = char_to_ix[letter]
  return tensor


class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size

    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(hidden_size, output_size)
    # forgot to init weights?
    initrange = 0.01
    self.i2h.weight.data.normal_(0, initrange)
    self.i2h.bias.data.fill_(0)
    self.i2o.weight.data.uniform_(0, initrange)
    self.i2o.bias.data.fill_(0)

  def forward(self, input, hidden):
    #print(input)
    #print(hidden)
    combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(hidden)
    return output, hidden

  def initHidden(self):
    return Variable(torch.zeros(1, self.hidden_size))

rnn = RNN(vocab_size, hidden_size, vocab_size)   
#for p in rnn.parameters():
#  p.data = torch.randn(p.data.size()) * 0.01

optimizer = torch.optim.Adagrad(rnn.parameters(), lr = learning_rate)  

# criterion = nn.NLLLoss()
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
  torch.nn.utils.clip_grad_norm(rnn.parameters(), 5.0, norm_type=1)
  optimizer.step()

  # Add parameters' gradients to their values, multiplied by learning rate
  #for p in rnn.parameters():
  #  p.data.add_(-learning_rate, p.grad.data)

  return loss.data[0]


p = 0
#mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
#mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
for iter in range(n_epoch + 1):

  if p+seq_length+1 >= len(data): p = 0

  inputs  = Variable(lineToTensor(data[p:p+seq_length]))
  targets = Variable(lineToLongTensor(data[p+1:p+seq_length+1]))
  loss = train(targets, inputs)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  # if smooth_loss > 60: smooth_loss = 60
  # Print iter number, loss, name and guess
  if iter % epoch_step == 0: print('iter %d, loss: %f' % (iter, smooth_loss))

  p += seq_length
