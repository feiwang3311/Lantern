from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope

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


def fully_connected(inputs, num_classes):
  # 1D batchNorm
  inputShape = tf.shape(inputs)
  inputs = tf.reshape(inputs, inputShape[0] * inputShape[1], inputShape[2])
  barch_norm = tf.layers.batch_normalization(inputs, axis=1, training=True, trainable=True)
  linear = tf.contrib.layers.fully_connected(batch_norm, num_classes, activation_fn=None)
  outputs = tf.reshape(linear, inputShape[0], inputShape[1], num_classes)
  return outputs


def BatchRNN(inputs, hidden_size):
  # 1D batchNorm
  inputShape = tf.shape(inputs)
  inputs = tf.reshape(inputs, inputShape[0] * inputShape[1], inputShape[2])
  barch_norm = tf.layers.batch_normalization(inputs, axis=1, training=True, trainable=True)
  inputs = tf.reshape(batch_norm, inputShape[0], inputShape[1], inputShape[2])

  # bidirectional RNN
  rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNTanh(1, hidden_size, direction=CUDNN_RNN_BIDIRECTION)
  # rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  # defining initial state
  # initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
  initial_state = rnn_cell._zero_state(batch_size, dtype=tf.float32)
  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, sequence_length=None,
                                     initial_state=initial_state, time_major=True,
                                     dtype=tf.float32)
  shape = tf.shape(outputs)
  outputs = tf.reshape(outputs, shape[0], shape[1], 2, -1)
  outputs = tf.math.reduce_sum(outputs, axis=2, keepdims=None)
  return outputs # checkshape


def inference(feats, sample_rate, window_size, rnn_hidden_size):
  # convolutions
  step1 = conv2d(feats, 32, [41, 11], stride=[2, 2], padding='VALID',
    data_format='NCHW', activation_fn=None, activation_fn=batch_norm,
    normalizer_params={'is_training': is_training,
                                    'fused': True,
                                    'decay': None})
  step2 = relux(step1, capping = 20)
  step3 = conv2d(step2, 32, [21, 11], stride=[2, 1], padding='VALID',
    data_format='NCHW', activation_fn=None, activation_fn=batch_norm,
    normalizer_params={'is_training': is_training,
                                    'fused': True,
                                    'decay': None})
  step4 = relux(step3, capping = 20)

  # reshape and transpose
  shape = tf.shape(step4)
  step5 = tf.reshape(step4, shape[0], shape[1] * shape[2], shape[3])
  step6 = tf.transpose(step5, perm=[2, 0, 1])

  # RNN layers
  # rnn_input_size = getInputSize(sample_rate, window_size)
  step7 = BatchRNN(step6, rnn_hidden_size)
  step8 = BatchRNN(step7, rnn_hidden_size)
  step9 = BatchRNN(step8, rnn_hidden_size)

  # fc layer
  step10 = fully_connected(step9)


def loss(feats, sample_rate, window_size, rnn_hidden_size, labels, percent):
  logits = inference(feats, sample_rate, window_size, rnn_hidden_size)
  # Calculate the average ctc loss across the batch.
  # labels: An int32 SparseTensor. labels.indices[i, :] == [b, t] means labels.values[i] stores the id for (batch b, time t).
  # labels.values[i] must take on values in [0, num_labels). See core/ops/ctc_ops.cc for more details.
  ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=tf.cast(logits, tf.float32),
                            sequence_length=tf.cast(tf.shape(logits)[0] * percent, tf.int32),
                            time_major=True)
  # ctc_loss = tf.Print(ctc_loss, [ctc_loss], "CTC loss: ", summarize=32)
  ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
  return ctc_loss_mean
