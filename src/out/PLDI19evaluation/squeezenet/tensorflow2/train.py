from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inputs
import squeezenet
import time
import tensorflow as tf

def train(args):
  startTime = time.time()
  x = tf.placeholder(tf.float32, shape = (args.batch_size, 3, 32, 32))
  y = tf.placeholder(tf.int32, shape = (args.batch_size))
  logits = squeezenet.Squeezenet_CIFAR1(args).build(x, is_training = True)
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=y, logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('optimizer'):
    train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(cross_entropy)

  batch = inputs.Batch(args.input_file, args.batch_size)

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

if __name__ == '__main__':
  # Training settings
  parser = argparse.ArgumentParser(description='TensorFlow cifar10 Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
            help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
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
  args = parser.parse_args()

  train(args)
