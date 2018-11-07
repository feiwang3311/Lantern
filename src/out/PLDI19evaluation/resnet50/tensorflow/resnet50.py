from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d, fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope

def conv3x3(inputs, num_outputs, stride = 1):
    return conv2d(inputs, num_outputs, [3, 3], stride=stride, data_format='NCHW', activation_fn=None, biases_initializer=None, scope = '3x3')

def bottleneck(inputs, planes, stride=1, downsample=None, expansion=4):
    conv1 = conv2d(inputs, planes, [1, 1], data_format='NCHW', normalizer_fn=batch_norm)                  # relu is default
    conv2 = conv2d(conv1, planes, [3, 3], data_format='NCHW', stride=stride, normalizer_fn=batch_norm)
    conv3 = conv2d(conv2, planes * expansion, [1, 1], data_format='NCHW', normalizer_fn=batch_norm, activation_fn=None)
    if downsample is not None:
        residual = downsample(inputs)
    else:
        residual = inputs
    out = conv3 + residual
    return tf.nn.relu(out)

def make_layer(inputs, block, planes, blocks, stride = 1, expansion = 4):
    inplanes = int(inputs.shape[1])
    def downsample(x):
        if stride != 1 or inplanes != planes * expansion:
            return conv2d(x, planes * expansion, [1, 1], data_format='NCHW', stride=stride, normalizer_fn=batch_norm, activation_fn=None)
        else:
            return x
    intermediate_step = block(inputs, planes, stride, downsample)
    for i in range(1, blocks):
        intermediate_step = block(intermediate_step, planes)
    return intermediate_step

def resnet50Cifar10(inputs, num_classes = 10):
    self_inplanes = 64
    conv1 = conv2d(inputs, 64, [3, 3], data_format='NCHW', stride=1, normalizer_fn=batch_norm)
    conv1_net = max_pool2d(conv1, [2, 2], data_format='NCHW', scope='maxpool1')
    conv2_net = make_layer(conv1_net, bottleneck, 64, 3)
    conv3_net = make_layer(conv2_net, bottleneck, 128, 4, stride=2)
    conv4_net = make_layer(conv3_net, bottleneck, 256, 6, stride=2)
    conv5_net = make_layer(conv4_net, bottleneck, 512, 3, stride=2)
    net = avg_pool2d(conv5_net, [2, 2], data_format='NCHW', scope='avgpool10')
    net = tf.reshape(net, [int(net.shape[0]), 512 * 4])
    return fully_connected(net, num_outputs=num_classes, activation_fn=None)
