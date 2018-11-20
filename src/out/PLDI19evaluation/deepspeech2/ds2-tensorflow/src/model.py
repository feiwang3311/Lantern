from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops import control_flow_ops
import math
import numpy as np

def relux(x, capping=None):
    """Clipped ReLU"""
    x = tf.nn.relu(x)
    if capping is not None:
        y = tf.minimum(x, capping)
    return y

def _variable_on_cpu(name, shape, initializer=None, use_fp16=False, trainable=True):
  with tf.device('/cpu'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_on_gpu(name, shape, initializer=None, use_fp16=False, trainable=True):
  with tf.device('/device:GPU:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def getInputSize(sample_rate, window_size):
  rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
  rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
  rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
  rnn_input_size *= 32
  return run_input_size

def getSeqLength(raw_length):
  seqL = tf.math.floor((raw_length - 11) / 2)
  seqL = seqL + 1 # tf.constant(1, dtype=tf.int32)
  seqL = seqL - 10 # tf.constant(10)
  return seqL

def batchNorm2D(input1, paramSize, variance_epsilon=0.00001):
  batch_mean, batch_var = tf.nn.moments(input1, [0, 2, 3], name='moments', keep_dims=True)
  ema = tf.train.ExponentialMovingAverage(decay=0.9997)
  def mean_var_with_update():
    ema_apply_op = ema.apply([batch_mean, batch_var])
    with tf.control_dependencies([ema_apply_op]):
      return tf.identity(batch_mean), tf.identity(batch_var)
  mean, var = control_flow_ops.cond(tf.constant(True), mean_var_with_update, lambda:(ema.average(batch_mean), ema.average(batch_var)))

  offset = _variable_on_gpu('offset', [1, paramSize, 1, 1], initializer=tf.zeros_initializer(), trainable=True)
  scale = _variable_on_gpu('scale', [1, paramSize, 1, 1], initializer=tf.ones_initializer(), trainable=True)
  return tf.nn.batch_normalization(input1, mean, var, offset, scale, variance_epsilon)


def batchNorm1D(input1, paramSize, variance_epsilon=0.00001):
  batch_mean, batch_var = tf.nn.moments(input1, [0], name='moments', keep_dims=True)
  ema = tf.train.ExponentialMovingAverage(decay=0.9997)
  def mean_var_with_update():
    ema_apply_op = ema.apply([batch_mean, batch_var])
    with tf.control_dependencies([ema_apply_op]):
      return tf.identity(batch_mean), tf.identity(batch_var)
  mean, var = control_flow_ops.cond(tf.constant(True), mean_var_with_update, lambda:(ema.average(batch_mean), ema.average(batch_var)))

  offset = _variable_on_gpu('offset', [paramSize], initializer=tf.zeros_initializer(), trainable=True)
  scale = _variable_on_gpu('scale', [paramSize], initializer=tf.ones_initializer(), trainable=True)

  return tf.nn.batch_normalization(input1, mean, var, offset, scale, variance_epsilon)


def fully_connected(inputs, batchSize, inputSize, num_classes):
  # 1D batchNorm
  with tf.variable_scope("batchNorm1D"):
    inputs = tf.reshape(inputs, [-1, inputSize])
    batch_norm = batchNorm1D(inputs, inputSize)
    batch_norm.set_shape([None, inputSize])
  with tf.variable_scope("fully_connected"):
    linear = tf.contrib.layers.fully_connected(batch_norm, num_classes, activation_fn=None)
    outputs = tf.reshape(linear, [-1, batchSize, num_classes])
  return outputs


def BatchRNN(inputs, batchSize, inputSize, hiddenSize):

  with tf.variable_scope("batchNorm1D"):
    inputs = tf.reshape(inputs, [-1, inputSize])
    batch_norm = batchNorm1D(inputs, inputSize)
    inputs = tf.reshape(batch_norm, [-1, batchSize, inputSize])
    inputs.set_shape([None, batchSize, inputSize])

  # bidirectional RNN
  # rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNTanh(1, hidden_size, direction='bidirectional')
  with tf.variable_scope("bidirectionalRNN"):
#    cell_fw = tf.nn.rnn_cell.BasicRNNCell(hiddenSize)
    cell_fw = tf.keras.layers.SimpleRNNCell(hiddenSize)
#    cell_bw = tf.nn.rnn_cell.BasicRNNCell(hiddenSize)
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

  with tf.variable_scope("convolution1"):
    step1 = conv2d(feats, 32, [41, 11], stride=[2, 2], padding='VALID',
      data_format='NCHW', activation_fn=None)
    # do batchNorm separately, otherwise it errors (if added in conv2d)
    step1 = batchNorm2D(step1, 32)
    step2 = relux(step1, capping = 20)

  with tf.variable_scope("convolution2"):
    step3 = conv2d(step2, 32, [21, 11], stride=[2, 1], padding='VALID',
      data_format='NCHW', activation_fn=None)
    step3 = batchNorm2D(step3, 32)
    step4 = relux(step3, capping = 20)

  # reshape and transpose
  with tf.variable_scope("reshape"):
    step5 = tf.reshape(step4, [32, 32 * 21, -1])
    step6 = tf.transpose(step5, perm=[2, 0, 1])
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
  with tf.variable_scope("fc"):
    step10 = fully_connected(step9, 32, rnn_hidden_size, num_classes)
  return step10


def loss(feats, sample_rate, window_size, rnn_hidden_size, labels, percent, raw_length, num_classes):
  logits = inference(feats, sample_rate, window_size, rnn_hidden_size, num_classes)
  # Calculate the average ctc loss across the batch.
  # labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the id for (batch b, time t).
  # labels.values[i] must take on values in [0, num_labels). See core/ops/ctc_ops.cc for more details.
  reducedLength = getSeqLength(raw_length)
  seqLength = tf.math.floor(reducedLength * percent)
  seqLength = tf.cast(seqLength, dtype=tf.int32)
 # pp = tf.print(seqLength)

  with tf.variable_scope("ctc_loss"):
#    with tf.control_dependencies([pp]):
    ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=logits,
                              sequence_length=seqLength,
                              preprocess_collapse_repeated=True,
                              time_major=True)
    # ctc_loss = tf.Print(ctc_loss, [ctc_loss], "CTC loss: ", summarize=32)
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
  return ctc_loss_mean
