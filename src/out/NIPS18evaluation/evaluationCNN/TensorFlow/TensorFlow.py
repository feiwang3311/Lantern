# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 10])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 10, 20])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([320, 50])
    b_fc1 = bias_variable([50])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 320])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([50, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def run(write_to):

  print("this is the start of reading data")
  startTime = time.time()

  # Import data
  mnist = input_data.read_data_sets(args.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('optimizer'):
    train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(cross_entropy)

  # with tf.name_scope('accuracy'):
  #   correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
  #   correct_prediction = tf.cast(correct_prediction, tf.float32)
  # accuracy = tf.reduce_mean(correct_prediction)

  #graph_location = tempfile.mkdtemp()
  #print('Saving graph to: %s' % graph_location)
  #train_writer = tf.summary.FileWriter(graph_location)
  #train_writer.add_graph(tf.get_default_graph())

  loopStart = time.time()
  loss_save = []
  with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
      train_accuracy = 0.0
      start = time.time() * 1000
      for i in range(60000 // args.batch_size):
        batch = mnist.train.next_batch(args.batch_size)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #print(loss)
        train_accuracy += loss
        #if (i + 1) % 60 == 0:
        #  print('epoch %d: step %d, training loss %f' % (epoch + 1, i + 1, train_accuracy / (i * 100)))
      stop = time.time() * 1000
      print('Training completed in {}ms ({}ms/image)'.format(int(stop - start), (stop - start)/60000))
      average_loss = train_accuracy / (60000 / args.batch_size)
      print('average loss is %s' % average_loss)
      loss_save.append(average_loss)

      #start = time.time() * 1000
      #tloss = 0
      #tacc = 0
      #for i in range(100):
      #  batch = mnist.test.next_batch(100)
      #  loss, acc = sess.run([cross_entropy, accuracy], feed_dict={
      #    x: batch[0], y_: batch[1], keep_prob: 1.0})
      #  tloss += loss
      #  tacc += acc
      #stop = time.time() * 1000

      #print('Epoch %d: test accuracy %d/10000. Average loss %f' % (epoch + 1, tacc, tloss / 10000))
      #print('Testing completed in {}ms ({}ms/image)'.format(int(stop - start), (stop - start)/10000))
  loopEnd = time.time()
  prepareTime = loopStart - startTime
  loopTime = loopEnd - loopStart
  timePerEpoch = loopTime / args.epochs

  with open(write_to, "w") as f:
    f.write("unit: " + "1 epoch\n")
    for loss in loss_save:
      f.write(str(loss) + "\n")
    f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")


if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
                      help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=6000, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--data_dir', type=str,
                     default='./input_data',
                     help='Directory for storing input data')
  args = parser.parse_args()

  import os
  if not os.path.exists(args.data_dir):
    # only try to download the data here
    input_data.read_data_sets(args.data_dir)

  run("result_TensorFlow"+str(args.batch_size)+".txt")