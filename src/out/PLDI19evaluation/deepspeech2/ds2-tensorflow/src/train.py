from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import user_defined_input
import model
import time
import tensorflow as tf
import statistics
import numpy as np

def train(args):
  print("run tensorflow deepspeech2")
  startTime = time.time()

  freq_size = 161
  batch_size = 32
  sample_rate = 16000
  window_size = 0.02
  rnn_hidden_size = 1024
  num_batches = 200
  x = tf.placeholder(tf.float32, shape=(batch_size, 1, freq_size, None), name="sequence_input")
  y = tf.sparse.placeholder(tf.int32, name="labels")
  percent = tf.placeholder(tf.float64, shape=(batch_size), name="percent_length")
  rawLength = tf.placeholder(tf.int32, shape=(1), name="max_length")

  num_classes = len(args.labels)  + 1
  ctc_loss = model.loss(x, sample_rate, window_size, rnn_hidden_size, y, percent, rawLength, num_classes)
  with tf.name_scope('optimizer'):
    train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(ctc_loss)

  filename = "/scratch/wu636/Lantern/src/out/PLDI19evaluation/deepspeech2/ds2-pytorch/data/test/deepspeech_train.pickle"
  batchedData = user_defined_input.Batch(filename)

  config = tf.ConfigProto()
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    loopStart = time.time()
    loss_save = []
    time_save = []
    for epoch in range(args.epochs):
      train_accuracy = 0.0
      start = time.time()
      for i in range(num_batches):
        inputs, targets, input_percentages, raw_length, target_sizes = batchedData.batchWithRawLength()
        # Need to process targets and target_size into SparseMatrix (i.e. indices, values, shape)
        values = targets
        ind = []
        for i_batch in range(batch_size):
          for d_batch in range(target_sizes[i_batch]):
            ind.append([i_batch, d_batch])
        indices = np.array(ind, dtype=np.int64)
        shape = np.array([batch_size, np.max(target_sizes)], dtype=np.int64)
        # indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
        # values = np.array([1.0, 2.0], dtype=np.float32)
        # shape = np.array([7, 9, 2], dtype=np.int64)
        _, loss = sess.run([train_step, ctc_loss], feed_dict={x: inputs, y: tf.SparseTensorValue(indices, values, shape), percent: input_percentages, rawLength: raw_length})
        train_accuracy += loss
        if (i + 1) % (20) == 0:
          print('epoch %d: step %d, training loss %f' % (epoch + 1, i + 1, train_accuracy / (i * 100)))
      stop = time.time()
      time_save.append(stop - start)
      average_loss = train_accuracy / (60000 / args.batch_size)
      print('Training completed in {}ms ({}ms/image), with average loss {}'.format((stop - start), (stop - start)/60000, average_loss))
      loss_save.append(average_loss)

  loopEnd = time.time()
  prepareTime = loopStart - startTime
  loopTime = loopEnd - loopStart
  timePerEpoch = loopTime / args.epochs

  time_save.sort()
  median_time = time_save[int (args.epochs / 2)]

  with open(args.write_to, "w") as f:
    f.write("unit: " + "1 epoch\n")
    for loss in loss_save:
      f.write(str(loss) + "\n")
    f.write("run time: " + str(prepareTime) + " " + str(median_time) + "\n")

if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='TensorFlow DeepSpeech2 Example')
  parser.add_argument('--batch-size', type=int, default=32, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=1, metavar='N',
            help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.0000005, metavar='LR',
            help='learning rate (default: 0.05)')
  parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
            help='SGD momentum (default: 0.5)')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--input_file', type=str,
           default='../../cifar10_data/cifar-10-batches-py/data_batch_1',
           help='Directory for storing input data')
  parser.add_argument('--write_to', type=str,
           default='result_TensorFlow',
           help='Directory for saving runtime performance')
  parser.add_argument('--batch_norm_decay', type=float, default=0.9)
  parser.add_argument('--weight_decay', type=float, default=0.0,
            help='''L2 regularization factor for convolution layer weights.
                    0.0 indicates no regularization.''')
  parser.add_argument('--labels', type=str, default = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
  args = parser.parse_args()

  train(args)
