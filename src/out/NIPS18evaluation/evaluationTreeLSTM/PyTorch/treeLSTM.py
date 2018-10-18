from __future__ import print_function

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import numpy as np
import torch.nn.functional as F


torch.set_num_threads(1)
startTime = time.time()
torch.manual_seed(7)
# read word embedding
word_embedding_size   = 300
word_embedding_file = "small_glove.txt"
word_embedding = []
with open(word_embedding_file, 'r') as f:
  for (counter, line) in enumerate(f):
    if counter == 0:
      word_embedding_length = int(line)
    else:
      word_embedding.append(np.asarray([float(i) for i in line.split()]).reshape(1, -1))

word_embedding = np.concatenate(word_embedding, axis = 0)
print(word_embedding.shape)
print(word_embedding_length)

# read tree_data
tree_data_file = "array_tree.txt"
scores = []
words = []
lchs = []
rchs = []
with open(tree_data_file, 'r') as f:
  for (counter, line) in enumerate(f):
    if counter == 0:
      tree_data_size = int(line)
    else:
      temp = np.asarray([int(i) for i in line.split()])
      if (counter-1) % 5 == 1: scores.append(temp)
      elif (counter-1) % 5 == 2: words.append(temp)
      elif (counter-1) % 5 == 3: lchs.append(temp)
      elif (counter-1) % 5 == 4: rchs.append(temp)
print(len(scores))
print(len(words))
print(len(lchs))
print(len(rchs))
print(tree_data_size)

# hyperparameters
hidden_size = 150
output_size = 5
learning_rate = 0.05
batch = 1  # using larger batch size actually hurt the performance


# parameters
# for leaf
scale = 0.01
class TreeNet(nn.Module):
  def __init__(self):
    super(TreeNet, self).__init__()
    self.Wi = nn.Parameter(torch.randn(hidden_size, word_embedding_size) * scale)
    self.bi = nn.Parameter(torch.randn(hidden_size) * scale)
    self.Wo = nn.Parameter(torch.randn(hidden_size, word_embedding_size) * scale)
    self.bo = nn.Parameter(torch.randn(hidden_size) * scale)
    self.Wu = nn.Parameter(torch.randn(hidden_size, word_embedding_size) * scale)
    self.bu = nn.Parameter(torch.randn(hidden_size) * scale)
    # for non leaf
    self.U0i = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U1i = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.bbi = nn.Parameter(torch.randn(hidden_size) * scale)
    self.U00f = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U01f = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U10f = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U11f = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.bbf = nn.Parameter(torch.randn(hidden_size) * scale)
    self.U0o = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U1o = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.bbo = nn.Parameter(torch.randn(hidden_size) * scale)
    self.U0u = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.U1u = nn.Parameter(torch.randn(hidden_size, hidden_size) * scale)
    self.bbu = nn.Parameter(torch.randn(hidden_size) * scale)
    # for softmax
    self.Why = nn.Parameter(torch.randn(output_size, hidden_size) * scale)
    self.by = nn.Parameter(torch.randn(output_size) * scale)

  # create a network for the xor problem given input and output
  def forward(self, scores, words, lchs, rchs):
    def rec(index):
      if (words[index] == -1):
        # branch node
        (l_loss, l_hidden, l_cell) = rec(lchs[index])
        (r_loss, r_hidden, r_cell) = rec(rchs[index])
        i_gate = torch.sigmoid(torch.matmul(self.U0i, l_hidden) + torch.matmul(self.U1i, r_hidden) + self.bbi)
        fl_gate = torch.sigmoid(torch.matmul(self.U00f, l_hidden) + torch.matmul(self.U01f, r_hidden) + self.bbf)
        fr_gate = torch.sigmoid(torch.matmul(self.U10f, l_hidden) + torch.matmul(self.U11f, r_hidden) + self.bbf)
        o_gate = torch.sigmoid(torch.matmul(self.U0o, l_hidden) + torch.matmul(self.U1o, r_hidden) + self.bbo)
        u_value = torch.tanh(torch.matmul(self.U0u, l_hidden) + torch.matmul(self.U1u, r_hidden) + self.bbu)
        cell = i_gate * u_value + fl_gate * l_cell + fr_gate * r_cell
        hidden = o_gate * torch.tanh(cell)
        logits = (torch.matmul(self.Why, hidden) + self.by).view(1, output_size)
        target = Var(torch.LongTensor([int(scores[index])]))
        loss = l_loss + r_loss + F.nll_loss(F.log_softmax(logits, dim=1), target)
        return (loss, hidden, cell)
      else:
        embedding_tensor = Var(torch.Tensor(word_embedding[words[index]]))
        i_gate = torch.sigmoid(torch.matmul(self.Wi, embedding_tensor) + self.bi)
        o_gate = torch.sigmoid(torch.matmul(self.Wo, embedding_tensor) + self.bo)
        u_value = torch.tanh(torch.matmul(self.Wu, embedding_tensor) + self.bu)
        cell = i_gate * u_value
        hidden = o_gate * torch.tanh(cell)
        logits = (torch.matmul(self.Why, hidden) + self.by).view(1, output_size)
        target = Var(torch.LongTensor([int(scores[index])]))
        loss = F.nll_loss(F.log_softmax(logits, dim=1), target)
        return (loss, hidden, cell)
    return rec(0)[0]

net = TreeNet()
opt = optim.Adagrad(net.parameters(), lr = learning_rate)

epocNum = 6
loopStart = time.time()
loss_save = []
for epoc in range(epocNum):
  total_loss = 0
  for n in range(tree_data_size):
    opt.zero_grad()
    loss = net.forward(scores[n], words[n], lchs[n], rchs[n])
    total_loss += loss.data[0]
    loss.backward()
    opt.step()
  loss_save.append(total_loss / tree_data_size)
  print("epoc {}, average_loss {}".format(epoc, total_loss / tree_data_size))

loopEnd = time.time()
print('looptime is %s ' % (loopEnd - loopStart))

prepareTime = loopStart - startTime
loopTime = loopEnd - loopStart
timePerEpoch = loopTime / epocNum

with open("result_PyTorch.txt", "w") as f:
  f.write("unit: " + "1 epoch\n")
  for loss in loss_save:
    f.write("{}\n".format(loss))
  f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")
