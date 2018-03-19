"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import time

# data I/O
start = time.time()
data = open('graham.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
end = time.time()
print("data loading time: %f" % (end - start))

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

start = time.time()
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
  if iter % epoch_step == 0: print('iter %d, loss: %f' % (iter, smooth_loss))
  p += seq_length
end = time.time()
print("training loop time: %f" % (end - start))
exit(0)


def prepare_sequence(seq, to_ix): # but this is only 1D 
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# Create the model:

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        #embeddings is a torch tensor, which can be initialized to pre-trained embeddings
        #embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
        #embedding.weight = nn.Parameter(embeddings)
        # OR
        #embed = nn.Embedding(num_embeddings, embedding_dim)
        #embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence): # sentence is 1D
        embeds = self.word_embeddings(sentence) # embeds might be 2D
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden) # view is forcing 3D, lstm_out should be 3D too
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # view is forcing 2D, sentence length is viewed as batch size now
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores # still 2D, batch_size * logProb

######################################################################
# Train the model:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)
