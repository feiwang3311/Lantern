import dynet as dy
import numpy as np
import time


def run(filename):

  startTime = time.time()

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
  m = dy.ParameterCollection()
  Wi = m.add_parameters((hidden_size, word_embedding_size), init='normal', std=0.01)
  bi = m.add_parameters(hidden_size, init = 0)
  Wo = m.add_parameters((hidden_size, word_embedding_size), init='normal', std=0.01)
  bo = m.add_parameters(hidden_size, init = 0)
  Wu = m.add_parameters((hidden_size, word_embedding_size), init='normal', std=0.01)
  bu = m.add_parameters(hidden_size, init = 0)
  # for non leaf
  U0i = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U1i = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  bbi = m.add_parameters(hidden_size, init = 0)
  U00f = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U01f = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U10f = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U11f = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  bbf = m.add_parameters(hidden_size, init = 0)
  U0o = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U1o = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  bbo = m.add_parameters(hidden_size, init = 0)
  U0u = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  U1u = m.add_parameters((hidden_size, hidden_size), init='normal', std=0.01)
  bbu = m.add_parameters(hidden_size, init = 0)
  # for softmax
  Why = m.add_parameters((output_size, hidden_size), init='normal', std=0.01)
  by = m.add_parameters(output_size, init = 0)

  trainer = dy.AdagradTrainer(m, learning_rate = learning_rate, eps = 1e-8)

  # create a network for the xor problem given input and output
  def tree_lstm_network(scores, words, lchs, rchs):
    def rec(index):
      if (words[index] == -1):
        # branch node
        (l_loss, l_hidden, l_cell) = rec(lchs[index])
        (r_loss, r_hidden, r_cell) = rec(rchs[index])
        i_gate = dy.logistic(U0i * l_hidden + U1i * r_hidden + bbi)
        fl_gate = dy.logistic(U00f * l_hidden + U01f * r_hidden + bbf)
        fr_gate = dy.logistic(U10f * l_hidden + U11f * r_hidden + bbf)
        o_gate = dy.logistic(U0o * l_hidden + U1o * r_hidden + bbo)
        u_value = dy.tanh(U0u * l_hidden + U1u * r_hidden + bbu)
        cell = dy.cmult(i_gate, u_value) + dy.cmult(fl_gate, l_cell) + dy.cmult(fr_gate, r_cell)
        hidden = dy.cmult(o_gate, dy.tanh(cell))
        pred1 = dy.log_softmax(Why * hidden + by)
        loss = l_loss + r_loss - pred1[int(scores[index])]
        return (loss, hidden, cell)
      else:
        embedding_tensor = dy.inputTensor(word_embedding[words[index]])
        i_gate = dy.logistic(Wi * embedding_tensor + bi)
        o_gate = dy.logistic(Wo * embedding_tensor + bo)
        u_value = dy.tanh(Wu * embedding_tensor + bu)
        cell = dy.cmult(i_gate, u_value)
        hidden = dy.cmult(o_gate, dy.tanh(cell))
        pred1 = dy.log_softmax(Why * hidden + by)
        loss = -pred1[int(scores[index])]
        return (loss, hidden, cell)
    return rec(0)[0]

  epocNum = 6
  loopStart = time.time()
  loss_save = []
  for epoc in range(epocNum):
    total_loss = 0
    for batch_n in range(int(tree_data_size // batch)):
      dy.renew_cg() # new computation graph
      losses = []
      for n in range(batch):
        index = batch_n * batch + n
        losses.append(tree_lstm_network(scores[index], words[index], lchs[index], rchs[index]))
        batch_loss = dy.esum(losses)
        total_loss += batch_loss.value()
        batch_loss.backward()
        trainer.update()
    loss_save.append(total_loss / tree_data_size)
    print("epoc {}, average_loss {}".format(epoc, total_loss / tree_data_size))

  loopEnd = time.time()
  print('looptime is %s ' % (loopEnd - loopStart))

  prepareTime = loopStart - startTime
  loopTime = loopEnd - loopStart
  timePerEpoch = loopTime / epocNum

  with open(filename, "w") as f:
    f.write("unit: " + "1 epoch\n")
    for loss in loss_save:
      f.write(str(loss) + "\n")
    f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")

  # --dynet-autobatch 1 --dynet-mem 2048

if __name__ == '__main__':
  import sys
  if (len(sys.argv) == 1):
    print("should have a file to write results to")
    exit(0)
  run(sys.argv[1])