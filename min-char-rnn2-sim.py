"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
vocab_size = 3
hidden_size = 2 
"""
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
"""
# import relevant supports
import torch
from torch.autograd import Variable
# have model parameters as globle variables, so the value is persistent
Wxh = torch.zeros((hidden_size, vocab_size)) + 0.01
Whh = torch.zeros((hidden_size, hidden_size)) + 0.01
Why = torch.zeros((vocab_size, hidden_size)) + 0.01
bh  = torch.zeros((hidden_size, 1))
by  = torch.zeros((vocab_size, 1))
"""
dWxh = torch.zeros((hidden_size, vocab_size))
dWhh = torch.zeros((hidden_size, hidden_size))
dWhy = torch.zeros((vocab_size, hidden_size))
dbh  = torch.zeros((hidden_size, 1))
dby  = torch.zeros((vocab_size, 1))
"""
# rewrite lossFun using Variables
def lossFunT(inputs, targets, hprev):
  """
    inputs targets are both list of integers
    hprev is Hx1 array of initial hidden state
    returns the loss
    update gradients on model parameter
    update last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = hprev
  loss = 0
  # set parameters as Variables here:
  Wxh1 = Variable(Wxh, requires_grad=True)
  Whh1 = Variable(Whh, requires_grad=True)
  Why1 = Variable(Why, requires_grad=True)
  bh1  = Variable(bh , requires_grad=True)
  by1  = Variable(by , requires_grad=True)
  for t in range(len(inputs)):
    temp = torch.zeros((vocab_size, 1)) # encode in 1-of-k representation
    temp[inputs[t]] = 1
    xs[t] = Variable(temp, requires_grad = False)
    hs[t] = torch.tanh(torch.mm(Wxh1, xs[t]) + torch.mm(Whh1, hs[t-1]) + bh1)
    ys[t] = torch.mm(Why1, hs[t]) + by1
    ps[t] = torch.exp(ys[t]) / torch.sum(torch.exp(ys[t]))
    loss += -torch.log(ps[t][targets[t], 0])
  # backward pass is simple
  loss.backward()
  # clip gradient [-5, 5] and copy to global variables
  #for cc, dd in zip([Wxh1, Whh1, Why1, bh1, by1],
  #                  [dWxh, dWhh, dWhy, dbh, dby]):
  #  cc.grad = torch.clamp(cc.grad, min = -5.0, max = 5.0)
  #  dd = cc.grad.data
  # update hprev
  return loss.data, Wxh1.grad, Whh1.grad, Why1.grad, bh1.grad, by1.grad, hs[len(inputs) - 1].data

result = lossFunT([0,1,2], [1,2,0], Variable(torch.zeros((hidden_size,1)), requires_grad=True))
loss, dWxh, dWhh, dWhy, dbh, dby, hid = result
print(loss)
print(dWxh)
print(dWhh)
print(dWhy)
print(dbh)
print(dby)
print(hid)
exit(0)


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


## rest program is not run
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
