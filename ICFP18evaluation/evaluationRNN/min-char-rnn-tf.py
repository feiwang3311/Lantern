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
    batch_size = 1

    # build model
    batchX_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])
    init_state = tf.placeholder(tf.float32, [batch_size, hidden_size])

    W2 = tf.Variable(np.random.randn(hidden_size, vocab_size) * learning_rate, dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,vocab_size)), dtype=tf.float32)

    # Unpack columns
    inputs_series = tf.unstack(tf.one_hot(batchX_placeholder, vocab_size), axis=1)
    # inputs_series = tf.split(axis=1, num_or_size_splits=seq_length, value=batchX_placeholder)
    labels_series = tf.unstack(batchY_placeholder, axis=1)

    # forward pass
    cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
    states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=init_state)
    logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
    # predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
    total_loss = tf.reduce_sum(losses)

    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    end = time.time()
    prepareTime = end - start
    #print("data loading time: %f" % (end - start))

    loss_save = []

    start = time.time()
    session_conf = tf.ConfigProto(
          intra_op_parallelism_threads=1,
          inter_op_parallelism_threads=1)
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
        loss_list = []
        p = 0
        for epoch_idx in range(num_iters + 1):
            if p+seq_length+1 >= len(data) or epoch_idx == 0: 
                p = 0 # go from start of data
                _current_state = np.zeros((1,hidden_size))

            inputs = np.array([char_to_ix[ch] for ch in data[p:p+seq_length]]).reshape((1,seq_length))
            targets = np.array([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]).reshape((1,seq_length))

            _total_loss, _train_step, _current_state, _states_series = sess.run(
                [total_loss, train_step, current_state, states_series],
                feed_dict={
                    batchX_placeholder:inputs,
                    batchY_placeholder:targets,
                    init_state:_current_state,
                })
            # print(_states_series)
            smooth_loss = smooth_loss * 0.9 + _total_loss * 0.1
            loss_list.append(smooth_loss)
            p += seq_length

            if epoch_idx % iter_step == 0:
                print("Step",epoch_idx, "Loss", smooth_loss)
                loss_save.append(smooth_loss)

    end = time.time()
    loopTime = end - start
#    print("training loop time: %f" % (end - start))

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