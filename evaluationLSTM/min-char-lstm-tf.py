# Python
"""
Adopted from the word-level language model(TensorFlow/tutorial/rnn/ptb).
Minimal character-level Vanilla RNN model. Written by Xilun Wu.
"""
import numpy as np
import tensorflow as tf
import time

# read file
start = time.time()
data = open('graham.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
end = time.time()
print("data loading time: %f" % (end - start))

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 20 # number of steps to unroll the RNN for
learning_rate = 1e-1
num_epochs = 5000
epoch_step = 250
batch_size = 1

# build model
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, seq_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
cell_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

W2 = tf.Variable(np.random.randn(hidden_size, vocab_size) * learning_rate,dtype=tf.float32)  #hidden to output
b2 = tf.Variable(np.zeros((1,vocab_size)), dtype=tf.float32)  # output bias

# Unpack columns
inputs_series = tf.split(axis=1, num_or_size_splits=seq_length, value=batchX_placeholder)  # [batch_size, seq_length] -> batch_size [1, seq_length] tensors
labels_series = tf.unstack(batchY_placeholder, axis=1)  # unpack the [batch_size, vocab_length] tensor into batch_szie [1, seq_length] tensors

# forward pass
cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, dtype=tf.float32)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_sum(losses)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

start = time.time()
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
with tf.Session(config=session_conf) as sess:
    sess.run(tf.initialize_all_variables())
    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    loss_list = []
    p = 0
    for epoch_idx in range(num_epochs + 1):
        if p+seq_length+1 >= len(data) or epoch_idx == 0: 
            p = 0 # go from start of data

        inputs = np.array([char_to_ix[ch] for ch in data[p:p+seq_length]]).reshape((1,seq_length))
        targets = np.array([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]).reshape((1,seq_length))

        _current_cell_state = np.zeros((batch_size, hidden_size))
        _current_hidden_state = np.zeros((batch_size, hidden_size))

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [total_loss, train_step, current_state, predictions_series],
            feed_dict={
                batchX_placeholder: inputs,
                batchY_placeholder: targets,
                cell_state: _current_cell_state,
                hidden_state: _current_hidden_state
        })

        _current_cell_state, _current_hidden_state = _current_state

        smooth_loss = smooth_loss * 0.9 + _total_loss * 0.1
        loss_list.append(smooth_loss)
        p += seq_length

        if epoch_idx%epoch_step == 0:
            print("Step",epoch_idx, "Loss", smooth_loss)
end = time.time()
print("training loop time: %f" % (end - start))
