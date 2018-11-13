"""
Custom RNN Cell definition.
Default RNNCell in TensorFlow throws errors when
variables are re-used between devices.
"""
import tensorflow as tf

from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.util import nest
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

from helper_routines import _variable_on_cpu

class CustomRNNCell(BasicRNNCell):
    """ This is a customRNNCell that allows the weights
    to be re-used on multiple devices. In particular, the Matrix of weights is
    set using _variable_on_cpu.
    The default version of the BasicRNNCell, did not support the ability to
    pin weights on one device (say cpu).
    """

    def __init__(self, num_units, input_size=None, activation=tf.nn.relu6, use_fp16=False):
        self._num_units = num_units
        self._activation = activation
        self.use_fp16 = use_fp16

    def __call__(self, inputs, state, scope = None):
        """Most basic RNN:
        output = new_state = activation(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):
            output = self._activation(_linear([inputs, state], self._num_units,
                                              True, use_fp16 = self.use_fp16))
        return output, output


class CustomRNNCell2(BasicRNNCell):
    """ This is a customRNNCell2 that allows the weights
    to be re-used on multiple devices. In particular, the Matrix of weights is
    set using _variable_on_cpu.
    The default version of the BasicRNNCell, did not support the ability to
    pin weights on one device (say cpu).
    """

    def __init__(self, num_units, input_size=None, activation=tf.nn.relu6):
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        """
         output = new_state = activation(BN(W * input) + U * state + B).
           state dim: batch_size * num_units
           input dim: batch_size * feature_size
           W: feature_size * num_units
           U: num_units * num_units
        """
        with tf.variable_scope(scope or type(self).__name__):
            # print "rnn cell input size: ", inputs.get_shape().as_list()
            # print "rnn cell state size: ", state.get_shape().as_list()
            wsize = inputs.get_shape()[1]
            w = _variable_on_cpu('W', [self._num_units, wsize], initializer=tf.orthogonal_initializer())
            # print w.name
            resi = tf.matmul(inputs, w, transpose_a=False, transpose_b=True)
            # batch_size * num_units
            bn_resi = seq_batch_norm(resi)
            # bn_resi = resi
            usize = state.get_shape()[1]
            u = _variable_on_cpu('U', [self._num_units, usize], initializer=tf.orthogonal_initializer())
            resu = tf.matmul(state, u, transpose_a=False, transpose_b=True)
            # res_nb = tf.add_n([bn_resi, resu])
            res_nb = tf.add(bn_resi, resu)
            bias = _variable_on_cpu('B', [self._num_units],
                                     tf.constant_initializer(0))
            res = tf.add(res_nb, bias)
            output = relux(res, capping=20)
        return output, output


def stacked_brnn(cell_fw, cell_bw, num_units, num_layers, inputs, batch_size, conved_seq_lens):
    """
    multi layer bidirectional rnn
    :param cell: RNN cell
    :param num_units: hidden unit of RNN cell
    :param num_layers: the number of layers
    :param inputs: the input sequence
    :param batch_size: batch size
    :return: the output of last layer bidirectional rnn with concatenating
    """
    prev_layer = inputs
    for i in xrange(num_layers):
        with tf.variable_scope("brnn-%d" % i) as scope:
            state_fw = cell_fw.zero_state(batch_size, tf.float32)
            state_bw = cell_fw.zero_state(batch_size, tf.float32)
            (outputs, state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, prev_layer, sequence_length=conved_seq_lens,
                                                               initial_state_fw=state_fw, initial_state_bw=state_bw, dtype=tf.float32, time_major=True) 
            outputs_fw, outputs_bw = outputs
            # prev_layer = tf.add_n([outputs_fw, outputs_bw])
            prev_layer = array_ops.concat(outputs, 2)
    return prev_layer


def relux(x, capping=None):
    """Clipped ReLU"""
    x = tf.nn.relu(x)
    if capping is not None:
        y = tf.minimum(x, capping)
    return y


def batch_norm2(inputs,
                decay=0.999,
                center=True,
                scale=True,
                epsilon=0.001,
                moving_vars='moving_vars',
                activation=None,
                is_training=True,
                trainable=True,
                scope=None,
                data_format='NHWC'):
  """Adds a Batch Normalization layer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    center: If True, subtract beta. If False, beta is not created and ignored.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    activation: activation function.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    a tensor representing the output of the operation.

  """
  inputs_shape = inputs.get_shape()
  with tf.variable_scope('bn2'):
    if data_format == 'NCHW':
      params_shape = inputs_shape[1]
    else:
      params_shape = inputs_shape[-1]

    # scale
    scale = _variable_on_cpu('scale', params_shape, initializer=tf.ones_initializer())
    # shift
    shift = _variable_on_cpu('shift', params_shape, initializer=tf.zeros_initializer())

    moving_mean = _variable_on_cpu('moving_mean', [params_shape], initializer=tf.zeros_initializer(), trainable=False)	
    moving_var = _variable_on_cpu('moving_variance', [params_shape], initializer=tf.ones_initializer(), trainable=False)

    if is_training:
      y, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=None, variance=None, epsilon=epsilon,
                                                        data_format=data_format, is_training=is_training)
      moving_averages.assign_moving_average(moving_mean, batch_mean, decay)
      moving_averages.assign_moving_average(moving_var, batch_var, decay)
    else:
      y, _, _ = tf.nn.fused_batch_norm(inputs, scale, shift, mean=moving_mean, variance=moving_var, epsilon=epsilon,
                                       data_format=data_format, is_training=is_training)  
    return y


def batch_norm(x, scope=None, is_train=True, data_format=None):
    """batch normalization, currently only work on NHWC"""
    with tf.variable_scope(scope or 'bn'):
        inputs_shape = x.get_shape()
        param_shape = inputs_shape[-1]        

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.9997)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(tf.cast(is_train, "bool"), mean_var_with_update, lambda:(ema.average(batch_mean), ema.average(batch_var)))

        offset = _variable_on_cpu('offset', [param_shape], initializer=tf.zeros_initializer())
        scale = _variable_on_cpu('scale', [param_shape], initializer=tf.ones_initializer())

        normed = tf.nn.batch_normalization(x, mean, var, offset, scale, 0.001)
    return normed


def seq_batch_norm(x, scope=None, is_train=True):
    """sequence batch normalization, input N * D"""
    with tf.variable_scope("sbn"):
        inputs_shape = x.get_shape()
        param_shape = inputs_shape[-1]

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.9997)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(tf.cast(is_train, "bool"), mean_var_with_update, lambda:(ema.average(batch_mean), ema.average(batch_var))) 

        offset = _variable_on_cpu('offset', [param_shape], initializer=tf.zeros_initializer(), trainable=True)
        scale = _variable_on_cpu('scale', [param_shape], initializer=tf.ones_initializer(), trainable=True)

        normed = tf.nn.batch_normalization(x, mean, var, offset, scale, 0.001)
    return normed


def _linear(args, output_size, bias, scope=None, use_fp16=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = _variable_on_cpu('Matrix', [total_arg_size, output_size],
                                  use_fp16 = use_fp16)
        if use_fp16:
            dtype = tf.float16
        else:
            dtype = tf.float32
        args = [tf.cast(x, dtype) for x in args]
        if len(args) == 1:
            res = tf.matmul(args[0], matrix, transpose_a=False, transpose_b=True)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix, transpose_a=False, transpose_b=True)
        if not bias:
            return res
        bias_term = _variable_on_cpu('Bias', [output_size],
                                     tf.constant_initializer(0),
                                     use_fp16=use_fp16)
    return res + bias_term
