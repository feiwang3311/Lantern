"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time

def run(write_to):
  # data I/O
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
  n_epoch = 5000
  epoch_step = 100

  # import relevant supports
  import torch
  import torch.nn as nn
  from torch.autograd import Variable
  import torch.nn.functional as F
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

  def lineToLongTensor1D(line):
    tensor = torch.LongTensor(len(line)).zero_()
    for li, letter in enumerate(line):
      tensor[li] = char_to_ix[letter]
    return tensor

  class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(RNN, self).__init__()

      self.hidden_size = hidden_size
      self.lstm = nn.LSTM(input_size, hidden_size)
  #    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
      self.i2o = nn.Linear(hidden_size, output_size)
      self.softmax = nn.LogSoftmax(dim=1)
      self.hidden = self.init_hidden()

    def forward(self, inputs):# inputs is 1D chars
  #    combined = torch.cat((input, hidden), 1)
  #    hidden = self.i2h(combined)
  #    output = self.i2o(hidden)
  #    output = self.softmax(output)
  #    return output, hidden

      inputsv = Variable(lineToTensor(inputs)) # inputsv is 3D
      lstm_out, self.hidden = self.lstm(inputsv, self.hidden)
      tag_space = self.i2o(lstm_out.view(len(inputs), -1))
      tag_scores = F.log_softmax(tag_space, dim=1)
      return tag_scores

    def init_hidden(self):
      #return Variable(torch.zeros(1, self.hidden_size))
      # The axes semantics are (num_layers, minibatch_size, hidden_dim)
      return (Variable(torch.zeros(1, 1, self.hidden_size)),
              Variable(torch.zeros(1, 1, self.hidden_size)))


  model = RNN(vocab_size, hidden_size, vocab_size)
  # loss_function = nn.NLLLoss()
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

  p = 0
  smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

  end = time.time()
  prepareTime = end - start
  #print("data loading time: %f" % (end - start))

  start = time.time()
  loss_save = []
  for iter in range(n_epoch + 1):
    if p+seq_length+1 >= len(data) or (iter == 0): 
      p = 0
      #model.hidden = model.init_hidden()  

    inputs  = data[p:p+seq_length]
    targets = data[p+1:p+seq_length+1]

    model.zero_grad()
    model.hidden = model.init_hidden()
    tag_scores = model(inputs)
    loss = loss_function(tag_scores, Variable(lineToLongTensor1D(targets)))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 5.0, norm_type=1)
    optimizer.step()

    smooth_loss = smooth_loss * 0.9 + loss.data[0] * 0.1
    if iter % epoch_step == 0: 
      print('iter %d, loss: %f' % (iter, smooth_loss))
      loss_save.append(smooth_loss)
    p += seq_length
  end = time.time()
  loopTime = end - start
  #print("training loop time: %f" % (end - start))
  
  with open(write_to, "w") as f:
    f.write("unit: " + "100 iteration\n")
    for loss in loss_save:
      f.write(str(loss) + "\n")
    f.write("run time: " + str(prepareTime) + " " + str(loopTime) + "\n")

if __name__ == '__main__':
  import sys
  if (len(sys.argv) != 2):
      print("should have a file to write results to")
      exit(0)
  run(sys.argv[1])