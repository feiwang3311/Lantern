from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
import math

def relux(x, capping=None):
    """Clipped ReLU"""
    x = tf.nn.relu(x)
    if capping is not None:
        y = tf.minimum(x, capping)
    return y


def getInputSize(sample_rate, window_size):
  rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
  rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
  rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
  rnn_input_size *= 32

def batchNorm2D(input1, paramSize, variance_epsilon=0.00001):
  mean   = tf.reshape(tf.Variable([0]*paramSize, dtype=tf.float32), [1, paramSize, 1, 1])
  var    = tf.reshape(tf.Variable([1]*paramSize, dtype=tf.float32), [1, paramSize, 1, 1])
  offset = tf.reshape(tf.Variable([0]*paramSize, dtype=tf.float32), [1, paramSize, 1, 1])
  scale  = tf.reshape(tf.Variable([1]*paramSize, dtype=tf.float32), [1, paramSize, 1, 1])
  return tf.nn.batch_normalization(input1, mean, var, offset, scale, variance_epsilon)

def batchNorm1D(input1, paramSize, variance_epsilon=0.00001):
  mean   = tf.reshape(tf.cast(tf.Variable([0]*paramSize), dtype=tf.float32), [1, paramSize])
  var    = tf.reshape(tf.cast(tf.Variable([1]*paramSize), dtype=tf.float32), [1, paramSize])
  offset = tf.reshape(tf.cast(tf.Variable([0]*paramSize), dtype=tf.float32), [1, paramSize])
  scale  = tf.reshape(tf.cast(tf.Variable([1]*paramSize), dtype=tf.float32), [1, paramSize])
  return tf.nn.batch_normalization(input1, mean, var, offset, scale, variance_epsilon)


def fully_connected(inputs, num_classes):
  # 1D batchNorm
  inputShape = tf.shape(inputs)
  inputs = tf.reshape(inputs, [inputShape[0] * inputShape[1], inputShape[2]])
  batch_norm = batchNorm1D(inputs, inputShape[2])
  # DANGER: (Fei Wang) hard code!!
  batch_norm.set_shape([None, 1024])
  linear = tf.contrib.layers.fully_connected(batch_norm, num_classes, activation_fn=None)
  outputs = tf.reshape(linear, [inputShape[0], inputShape[1], num_classes])
  return outputs


def BatchRNN(inputs, batchSize, inputSize, hiddenSize):

  # 1D batchNorm
  inputShape = tf.shape(inputs)
  inputs = tf.reshape(inputs, [inputShape[0] * inputShape[1], inputShape[2]])
  batch_norm = batchNorm1D(inputs, inputShape[2])
  inputs = tf.reshape(batch_norm, [inputShape[0], inputShape[1], inputShape[2]])
  inputs.set_shape([None, batchSize, inputSize])
  print("inputs size is {}".format(inputs.shape))

  # bidirectional RNN
  # rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNTanh(1, hidden_size, direction='bidirectional')
  # Need to optimize
  cell_fw = tf.keras.layers.SimpleRNNCell(hiddenSize)
  cell_bw = tf.keras.layers.SimpleRNNCell(hiddenSize)
  # initial_state = rnn_cell._zero_state(inputShape[1])
  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
  # 'state' is a tensor of shape [batch_size, cell_state_size]
  ((outputs_fw, outputs_bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                        cell_bw, inputs, dtype=tf.float32,
                                        time_major=True)

  #outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, sequence_length=None,
  #                                   initial_state=initial_state, time_major=True,
  #                                   dtype=tf.float32)
  # shape = tf.shape(outputs)
  # outputs = tf.reshape(outputs, [shape[0], shape[1], 2, -1])
  # outputs = tf.math.reduce_sum(outputs, axis=2, keepdims=None)
  outputs = outputs_fw + outputs_bw
  return outputs # checkshape


def inference(feats, sample_rate, window_size, rnn_hidden_size, num_classes):

  # convolution1 
  step1 = conv2d(feats, 32, [41, 11], stride=[2, 2], padding='VALID',
    data_format='NCHW', activation_fn=None)
  # do batchNorm separately, otherwise it errors (if added in conv2d)
  step1 = batchNorm2D(step1, step1.shape[1])
  step2 = relux(step1, capping = 20)

  # convolution2
  step3 = conv2d(step2, 32, [21, 11], stride=[2, 1], padding='VALID',
    data_format='NCHW', activation_fn=None)
  step3 = batchNorm2D(step3, step1.shape[1])
  step4 = relux(step3, capping = 20)
  print("step4 {}".format(step4.shape))

  # reshape and transpose
  shape = tf.shape(step4)
  step5 = tf.reshape(step4, [shape[0], shape[1] * shape[2], shape[3]])
  step6 = tf.transpose(step5, perm=[2, 0, 1])
  # DANGER: (Fei Wang) hard code!!
  step6.set_shape([None, 32, 32 * 21])
  
  # RNN layers
  # rnn_input_size = getInputSize(sample_rate, window_size)
  with tf.variable_scope("BatchRNN1"):
    step7 = BatchRNN(step6, 32, 32 * 21, rnn_hidden_size)
  with tf.variable_scope("BatchRNN2"):
    step8 = BatchRNN(step7, 32, rnn_hidden_size, rnn_hidden_size)
  with tf.variable_scope("BatchRNN3"):
    step9 = BatchRNN(step8, 32, rnn_hidden_size, rnn_hidden_size)

  # fc layer
  step10 = fully_connected(step9, num_classes)


def loss(feats, sample_rate, window_size, rnn_hidden_size, labels, percent, num_classes):
  logits = inference(feats, sample_rate, window_size, rnn_hidden_size, num_classes)
  # Calculate the average ctc loss across the batch.
  # labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the id for (batch b, time t).
  # labels.values[i] must take on values in [0, num_labels). See core/ops/ctc_ops.cc for more details.
  ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits,
                            sequence_length=tf.cast(tf.shape(logits)[0] * percent, tf.int32),
                            time_major=True)
  # ctc_loss = tf.Print(ctc_loss, [ctc_loss], "CTC loss: ", summarize=32)
  ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
  return ctc_loss_mean
