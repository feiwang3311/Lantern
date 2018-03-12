# Python
"""
Adopted from the word-level language model(TensorFlow/tutorial/rnn/ptb).
Minimal character-level Vanilla RNN model. Written by Xilun Wu.
"""
import numpy as np
import tensorflow as tf
"""
def _build_vocab(filename=None):
    data = open(filename, 'r').read() # should be simple plain text file
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = open(filename, 'r').read()
    return [word_to_id[word] for word in data if word in word_to_id]

def _raw_data():
    filename = "input.txt"
    word_to_id = _build_vocab(filename)
    train_data = _file_to_word_ids(filename, word_to_id)
    vocab_size = len(word_to_id)
    return train_data, vocab_size

def data_producer(raw_data, batch_size, num_steps):
    with tf.name_scope(None, "DataProducer", [raw_data, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
        # num_steps is the length of vector input for each iteration(epoch).
        epoch_size = (batch_len - 1) // num_steps  # epoch_size is the number of iterations in training

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
"""
# read file
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 50 # size of hidden layer of neurons
seq_length = 20 # number of steps to unroll the RNN for
<<<<<<< HEAD
learning_rate = 1e-2
=======
learning_rate = 1e-1
>>>>>>> 8b3e101506936d705155ab5b122bd3be3a27755b
num_epochs = 2001
batch_size = 1

# build model
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, seq_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
init_state = tf.placeholder(tf.float32, [batch_size, hidden_size])

W2 = tf.Variable(np.random.randn(hidden_size, vocab_size) * learning_rate,dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,vocab_size)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(axis=1, num_or_size_splits=seq_length, value=batchX_placeholder)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# forward pass
# cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)
# state_series = []
# for input in inputs_series:
#     output, state = cell(input, init_state[0])
#     state_series.append(output)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_sum(losses)
<<<<<<< HEAD
# total_loss = tf.reduce_mean(losses)
=======
>>>>>>> 8b3e101506936d705155ab5b122bd3be3a27755b

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    p = 0
    for epoch_idx in range(num_epochs):
        if p+seq_length+1 >= len(data) or epoch_idx == 0: 
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            p = 0 # go from start of data
            inputs = np.array([char_to_ix[ch] for ch in data[p:p+seq_length]]).reshape((1,seq_length))
            targets = np.array([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]).reshape((1,seq_length))

        _current_state = np.zeros((1,hidden_size))

        # print("New data, epoch", epoch_idx)

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [total_loss, train_step, current_state, predictions_series],
            feed_dict={
                batchX_placeholder:inputs,
                batchY_placeholder:targets,
                init_state:_current_state
            })

        loss_list.append(_total_loss)

        if epoch_idx%100 == 0:
            print("Step",epoch_idx, "Loss", _total_loss)
