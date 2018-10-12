# Python
"""
Adopted from the word-level language model(TensorFlow/tutorial/rnn/ptb).
Minimal character-level Vanilla RNN model. Written by Xilun Wu.
"""
import numpy as np
import tensorflow as tf
import time

def run(write_to):
  # read file
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
  num_iters = 5000
  iter_step = 100
  batch_size = 20

  # build model
  batchX_placeholder = tf.placeholder(tf.int32, [seq_length, batch_size])
  batchY_placeholder = tf.placeholder(tf.int32, [seq_length, batch_size])
  cell_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
  hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
  init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

  W2 = tf.Variable(np.random.randn(hidden_size, vocab_size) * 0.01, dtype=tf.float32)  #hidden to output
  b2 = tf.Variable(np.zeros((1,vocab_size)), dtype=tf.float32)  # output bias

  outputs, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True), tf.one_hot(batchX_placeholder, vocab_size), time_major=True, dtype = tf.float32)
  outputs = tf.reshape(outputs, [-1, hidden_size])
  Y = tf.reshape(batchY_placeholder, [-1])
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = (tf.matmul(outputs, W2) + b2), labels = Y)
  total_loss = tf.reduce_sum(loss)

  train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

  end = time.time()
  prepareTime = end - start

  loss_save = []

  start = time.time()
  session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
  with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())
    p = -seq_length * batch_size
    for epoch_idx in range(num_iters + 1):
      p += seq_length * batch_size
      if p + seq_length*batch_size + 1 >= len(data) or epoch_idx == 0:
        p = 0 # go from start of data

      inputs = np.transpose(np.array([char_to_ix[ch] for ch in data[p:p+seq_length*batch_size]]).reshape((batch_size, seq_length)))
      targets = np.transpose(np.array([char_to_ix[ch] for ch in data[p+1:p+seq_length*batch_size+1]]).reshape((batch_size, seq_length)))

      _total_loss, _train_step = sess.run([total_loss, train_step],
        feed_dict={
          batchX_placeholder: inputs,
          batchY_placeholder: targets,
      })

      if epoch_idx%iter_step == 0:
        print("Step",epoch_idx, "Loss", _total_loss)
        loss_save.append(_total_loss)

  end = time.time()
  loopTime = end - start

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