"""
Custom RNN Cell definition.
Default RNNCell in TensorFlow throws errors when
variables are re-used between devices.
"""
import tensorflow as tf

from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.util import nest
from tensorflow.python.training import moving_averages

from tensorflow.contrib.mkldnn_rnn.python.ops import mkldnn_rnn_ops

from helper_routines import _variable_on_cpu

class MkldnnRNNCell(BasicRNNCell):
    """ This is a MkldnnRNNCell based on MKLDNN engine. The Matrix of weights is
    set using _variable_on_cpu.
    The default version of the BasicRNNCell, did not support the ability to
    pin weights on one device (say cpu).
    """
    def __init__(self, sess, num_units, input_size = None, activation=tf.nn.relu6, use_fp16=False):
        self._num_units = num_units
        self.use_fp16 = use_fp16
        self.model = mkldnn_rnn_ops.MkldnnRNNRelu(1, self._num_units, input_size, dropout=0.0)
        param_size_t = self.model.params_size()
        if sess is not None:
          self.param_size = sess.run(param_size_t)
        # print "param size: ", self.param_size

    def __call__(self, inputs, state, scope=None, weight_size=None):
        with tf.variable_scope(scope or type(self).__name__):
          # if len(inputs.get_shape()) == 2:
          #   inputs = tf.expand_dims(inputs, axis=0)
          # state = tf.expand_dims(state, axis=0)
          # print "input size: ", inputs.get_shape(), " state size: ", state.get_shape()
          rnn_weights = _variable_on_cpu("rnn_weights", [self.param_size], tf.constant_initializer(1.0 / self.param_size), self.use_fp16)
          output, output_h = self.model(input_data=inputs,
                                        input_h=state,
                                        params=rnn_weights,
                                        is_training=True)
          # print "output size: ", output.get_shape(), "output h size: ", output_h.get_shape()
          # output = tf.squeeze(output, axis=0)
          # output_h = tf.squeeze(output_h, axis=0)
        return output, output_h
