from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inputs
import time
import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare


def train(args):
  startTime = time.time()
  start = startTime
  model = onnx.load('../squeezenetCifar10.onnx')
  tf_rep = prepare(model)
  tf_rep.graph.as_default()
  # TODO set tensorflow to use 1 thread ???
  # TODO: how to control using GPU or not using GPU?? it seems that using GPU is the default for TensorFlow
  batch = inputs.Batch(args.input_file, args.batch_size)

  def train_epoch(epoch):
    graph_input = tf_rep.tensor_dict[tf_rep.inputs[0]]
    graph_output = tf_rep.tensor_dict[tf_rep.outputs[0]]
    x = graph_input
    logits = graph_output
    y = tf.placeholder(tf.int32, shape = (args.batch_size))
    with tf.name_scope('loss'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
      cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope('optimizer'):
      train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(cross_entropy)

    loopStart = time.time()
    loss_save = []
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for epoch in range(args.epochs):
        train_accuracy = 0.0
        start = time.time() * 1000
        for i in range(batch.total_size // batch.batch_size):
          (input_x, input_y) = batch.batch()
          _, loss = sess.run([train_step, cross_entropy], feed_dict={x: input_x, y: input_y})
          train_accuracy += loss
          if (i + 1) % (batch.total_size // batch.batch_size // 10) == 0:
            print('epoch %d: step %d, training loss %f' % (epoch + 1, i + 1, train_accuracy / (i * 100)))
        stop = time.time() * 1000
        print('Training completed in {}ms ({}ms/image)'.format(int(stop - start), (stop - start)/60000))
        average_loss = train_accuracy / (60000 / args.batch_size)
        print('average loss is %s' % average_loss)
        loss_save.append(average_loss)

    loopEnd = time.time()
    prepareTime = loopStart - startTime
    loopTime = loopEnd - loopStart
    timePerEpoch = loopTime / args.epochs

    with open(args.write_to, "w") as f:
      f.write("unit: " + "1 epoch\n")
      for loss in loss_save:
        f.write(str(loss) + "\n")
      f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")

  def inference_epoch(epoch):
    for i in range(batch.total_size // batch.batch_size):
      (input_x, input_y) = batch.batch()
      tf_rep.run(input_x)._0
    return 0

  loopStart = time.time()
  loss_save = []

  if args.inference:
    for epoch in range(args.epochs):
      loss_save.append(inference_epoch(epoch))
      stop = time.time()
      print('Inferencing completed in {} sec ({} sec/image)'.format((stop - start), (stop - start)/60000))
    loopEnd = time.time()
    prepareTime = loopStart - startTime
    loopTime = loopEnd - loopStart
    timePerEpoch = loopTime / args.epochs
    with open(args.write_to, "w") as f:
      f.write("unit: " + "1 epoch\n")
      for loss in loss_save:
        f.write("{}\n".format(loss))
      f.write("run time: " + str(prepareTime) + " " + str(timePerEpoch) + "\n")
  else:
    train_epoch(0)

if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
            help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
            help='learning rate (default: 0.005)')
  parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
            help='SGD momentum (default: 0.0)')
  parser.add_argument('--seed', type=int, default=42, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--input_file', type=str,
           default='../../cifar10_data/cifar-10-batches-py/data_batch_1',
           help='Directory for storing input data')
  parser.add_argument('--write_to', type=str,
           default='result_PyTorch',
           help='Directory for saving performance data')
  parser.add_argument('--use_gpu', type=bool, default=False,
           help='Set to true if you want to use GPU')
  parser.add_argument('--inference', type=bool, default=False,
           help='Set to false if you want to measure inference time')
  args = parser.parse_args()

  train(args)
