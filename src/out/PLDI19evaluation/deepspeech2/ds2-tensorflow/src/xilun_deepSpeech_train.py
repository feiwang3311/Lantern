# Author: Lakshmi Krishnan
# Email: lkrishn7@ford.com
# Author: YAO Matrix
# Email: yaoweifeng0301@126.com


"""A script to train a deepSpeech model on LibriSpeech data.

References:
1. Hannun, Awni, et al. "Deep speech: Scaling up end-to-end
speech recognition." arXiv preprint arXiv:1412.5567 (2014).
2. Amodei, Dario, et al. "Deep speech 2: End-to-end
 speech recognition in english and mandarin."
 arXiv preprint arXiv:1512.02595 (2015).
"""

from datetime import datetime
import os.path
import re
import time
import argparse
import json
import sys
import numpy as np
import distutils.util

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug

import helper_routines
from setenvs import setenvs
from setenvs import arglist

def parse_args():
    " Parses command line arguments."
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        default='../models/librispeech/train',
                        help='Directory to write event logs and checkpoints')
    parser.add_argument('--platform', type=str,
                        default='knl',
                        help='running platform: knl or bdw')
    parser.add_argument('--data_dir', type=str,
                        default='',
                        help='Path to the audio data directory')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Number of batches to run')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of inputs to process in a batch per GPU')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle', dest='shuffle',
                                action='store_true')
    feature_parser.add_argument('--no-shuffle', dest='shuffle',
                                action='store_false')
    parser.set_defaults(shuffle=True)

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--use_fp16', dest='use_fp16',
                                action='store_true')
    feature_parser.add_argument('--use_fp32', dest='use_fp16',
                                action='store_false')
    parser.set_defaults(use_fp16=False)

    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')
    parser.add_argument('--num_hidden', type=int, default=1024,
                        help='Number of hidden nodes')
    parser.add_argument('--num_rnn_layers', type=int, default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Continue training from checkpoint file')
    # birectional RNN or unidirectional???
    parser.add_argument('--rnn_type', type=str, default='bidirectional',
                        help='unidirectional or bidirectional')
    parser.add_argument('--initial_lr', type=float, default=2e-5,
                        help='Initial learning rate for training')
    parser.add_argument('--num_filters', type=int, default=32,
                        help='Number of convolutional filters')
    parser.add_argument('--moving_avg_decay', type=float, default=0.9999,
                        help='Decay to use for the moving average of weights')
    parser.add_argument('--num_epochs_per_decay', type=int, default=5,
                        help='Epochs after which learning rate decays')
    parser.add_argument('--lr_decay_factor', type=float, default=0.999,
                        help='Learning rate decay factor')
    parser.add_argument('--intra_op', type=int, default=44,
                        help='Intra op thread num')
    parser.add_argument('--inter_op', type=int, default=1,
                        help='Inter op thread num')
    parser.add_argument('--engine', type=str, default='tf',
                        help='Select the engine you use: tf, mkl, mkldnn_rnn, cudnn_rnn')
    parser.add_argument('--debug', type=distutils.util.strtobool, default=False,
                        help='Switch on to enable debug log')
    parser.add_argument('--nchw', type=distutils.util.strtobool, default=True,
                        help='Whether to use nchw memory layout')
    parser.add_argument('--dummy', type=distutils.util.strtobool, default=False,
                        help='Whether to use dummy data rather than librispeech data')
 
    args = parser.parse_args()

    print "debug: ", args.debug
    print "nchw: ", args.nchw
    print "dummy: ", args.dummy
    print "engine: ", args.engine
    print "initial lr: ", args.initial_lr

    # Read architecture hyper-parameters from checkpoint file
    # if one is provided.
    if args.checkpoint is not None:
        param_file = os.path.join(args.checkpoint, 'deepSpeech_parameters.json')
        with open(param_file, 'r') as file:
            params = json.load(file)
            # Read network architecture parameters from previously saved
            # parameter file.
            args.num_hidden = params['num_hidden']
            args.num_rnn_layers = params['num_rnn_layers']
            args.rnn_type = params['rnn_type']
            args.num_filters = params['num_filters']
            args.use_fp16 = params['use_fp16']
            args.initial_lr = params['initial_lr']
            args.engine = params['engine']
    return args

ARGS = parse_args()

import deepSpeech

g = tf.Graph()
profiling = []

def tower_loss(sess, feats, labels, seq_lens):
    """Calculate the total loss on a single tower running the deepSpeech model.

    This function builds the graph for computing the loss per tower(GPU).

    ARGS:
      feats: Tensor of shape BxFxT representing the
             audio features (mfccs or spectrogram).
      labels: sparse tensor holding labels of each utterance.
      seq_lens: tensor of shape [batch_size] holding
              the sequence length per input utterance.
    Returns:
       Tensor of shape [batch_size] containing
       the total loss for a batch of data
    """

    # Build inference Graph.
    logits = deepSpeech.inference(sess, feats, seq_lens, ARGS)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    total_loss = deepSpeech.loss(logits, labels, seq_lens)

    # Compute the moving average of all individual losses and the total loss.
    # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # loss_averages_op = loss_averages.apply([total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    # loss_name = total_loss.op.name
    # Name each loss as '(raw)' and name the moving average
    # version of the loss as the original loss name.
    # tf.summary.scalar(loss_name + '(raw)', total_loss)

    # Without this loss_averages_op would never run
    # with tf.control_dependencies([loss_averages_op]):
    #     total_loss = tf.identity(total_loss)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the
       gradient has been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for each_grad, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(each_grad, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # The variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        weights = grad_and_vars[0][1]
        grad_and_var = (grad, weights)
        average_grads.append(grad_and_var)
    return average_grads


def set_learning_rate():
    """ Set up learning rate schedule """

    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed.
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (deepSpeech.NUM_PER_EPOCH_FOR_TRAIN / ARGS.batch_size)
    decay_steps = int(num_batches_per_epoch * ARGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(ARGS.initial_lr,
                                               global_step,
                                               decay_steps,
                                               ARGS.lr_decay_factor,
                                               staircase=True)

    return learning_rate, global_step


def fetch_data():
    """ Fetch features, labels and sequence_lengths from a common queue."""
    tot_batch_size = ARGS.batch_size 
    with tf.device('/device:GPU:0'):
        feats, labels, seq_lens = deepSpeech.inputs(eval_data='train',
                                                    data_dir=ARGS.data_dir,
                                                    batch_size=tot_batch_size,
                                                    use_fp16=ARGS.use_fp16,
                                                    shuffle=ARGS.shuffle)
        dense_labels = tf.sparse_tensor_to_dense(labels)
#        tf.Print(dense_labels, [dense_labels], "labels")

    # Split features and labels and sequence lengths for each tower
    return feats, labels, seq_lens


def get_loss_grads(sess, data, optimizer):
    """ Set up loss and gradient ops.
    Add summaries to trainable variables """

    # Calculate the gradients
    # data loading is executed here.
    [feats, labels, seq_lens] = data
    
    grads_and_vars = None
    with tf.device('/device:GPU:0'):
        # Calculate the loss for the deepSpeech model.
        loss = tower_loss(sess, feats, labels, seq_lens)

        # Retain the summaries from the final tower.
        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Calculate the gradients for the batch of data.
        grads_and_vars = optimizer.compute_gradients(loss)

        # Clip the gradients.
        clipped_grads_and_vars = [(tf.clip_by_value(grad, clip_value_min=-400, clip_value_max=400), var) for grad, var in grads_and_vars]

    return loss, clipped_grads_and_vars


def run_train_loop(sess, operations):
    """ Train the model for required number of steps."""
    (loss_op, train_op) = operations
    num_batches_per_epoch = (deepSpeech.NUM_PER_EPOCH_FOR_TRAIN / ARGS.batch_size)
    run_options = None
    run_metadata = None
    trace_file = None
    if ARGS.debug:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        trace_file = open('profiling.json', 'w')

    # Evaluate the ops for max_steps
    epoch = 0
    for step in range(ARGS.max_steps):
        if step % num_batches_per_epoch == 0:
            epoch = epoch + 1
        batch_time_start = time.time()

        # print "Trainable Variables: "
        # tvariables_names = [v.name for v in tf.trainable_variables()]
        # tvalues = sess.run(tvariables_names)
        # for k, v in zip(tvariables_names, tvalues):
        #     print "Variable: ", k
        #     print v
        # print "Global Variables: "
        # gvariables_names = [v.name for v in tf.global_variables()]
        # gvalues = sess.run(gvariables_names)
        # for k, v in zip(gvariables_names, gvalues):
        #     print "Variable: ", k
        #     print v
        # print "Moving Average Variables: "
        # mvariables_names = [v.name for v in tf.moving_average_variables()]
        # mvalues = sess.run(mvariables_names)
        # for k, v in zip(mvariables_names, mvalues):
        #     print "Variable: ", k
        #     print v
        forward_time_start = time.time()

        loss_value = sess.run(loss_op, options=run_options, run_metadata=run_metadata)
        
        forward_time = time.time() -forward_time_start

        backward_time_start = time.time()

        sess.run(train_op, options=run_options, run_metadata=run_metadata)

        backward_time = time.time() - backward_time_start

        batch_time = time.time() - batch_time_start
        
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        # separate data loading time;
        # modify the formatted string;
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {3:.3f}\t'
              'Forward Time {4:.3f}\t'
              'Backward Time {5:.3f}\t'
              'Loss {6:.4f}\t'.format(
            epoch, step % num_batches_per_epoch + 1, num_batches_per_epoch, batch_time,
            forward_time, backward_time, loss_value))

        """
        if step >= 10:
          profiling.append(duration)

        # tf.clear_collection("losses")

        # Print progress periodically
        if step > 10 and step % 10 == 0:
            examples_per_sec = (ARGS.batch_size * 1) / np.average(profiling)
            format_str = ('%s: step %d, '
                          'loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                  examples_per_sec, np.average(profiling) / 1))

        # Run the summary ops periodically
        if 0:
            summary_writer = tf.summary.FileWriter(ARGS.train_dir, sess.graph)
            summary_writer.add_summary(sess.run(summary_op), step)

        # Save the model checkpoint periodically
        if step % 20000 == 0 or (step + 1) == ARGS.max_steps:
            checkpoint_path = os.path.join(ARGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        if ARGS.debug and step == 20:
            trace = timeline.Timeline(run_metadata.step_stats)
            trace_file.write(trace.generate_chrome_trace_format())

            prof_options = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
            prof_options['output'] = "file:outfile=./params.log"
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                                tfprof_options=prof_options)
            # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

            prof_options = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS
            prof_options['output'] = "file:outfile=./flops.log"
            tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                  run_meta=run_metadata,
                                                                  tfprof_options=prof_options)

            prof_options = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY
            prof_options['output'] = "file:outfile=./timing_memory.log"
            prof_options['start_name_regexes'] = "ctc_loss"
            tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                  tfprof_cmd='graph',
                                                                  run_meta=run_metadata,
                                                                  tfprof_options=prof_options)
        """


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""
    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(ARGS.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        checkpoint_path = ckpt.model_checkpoint_path
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        print('No checkpoint file found')
        return

"""
def add_summaries(summaries, learning_rate, grads):

    # Track quantities for Tensorboard display
    summaries.append(tf.summary.scalar('learning_rate', learning_rate))
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)
    return summary_op
"""

def train():
    """
    Train deepSpeech for a number of steps.
      This function build a set of ops required to build the model and optimize
      weights.
    """
    with g.as_default(), tf.device('/device:GPU:0'):
        # Learning rate set up
        learning_rate, global_step = set_learning_rate()

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.keras.optimizers.SGD(learning_rate,
        #                                    momentum=0.9,
        #                                    decay=ARGS.lr_decay_factor)


        # Fetch a batch worth of data
        data = fetch_data()

        # Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the
        # ops do not have GPU implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=ARGS.log_device_placement))

        # Construct loss and gradient ops
        loss_op, grads = get_loss_grads(sess, data, optimizer)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads,
                                                      global_step=global_step)

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(ARGS.moving_avg_decay, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        # train_op = tf.group(apply_gradient_op, variables_averages_op)

        train_op = apply_gradient_op

        # Build summary op
        # summary_op = add_summaries(summaries, learning_rate, grads)

        # Create a saver.
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Initialize vars.
        """
        if ARGS.checkpoint is not None:
            print "can use checkpoint"
            global_step = initialize_from_checkpoint(sess, saver)
        else:
            print "cannot use checkpoint"
            sess.run(tf.global_variables_initializer())
        """
        print "forbid the use of checkpoint"
        sess.run(tf.global_variables_initializer())
        # print "Trainable Variables: "
        # tvariables_names = [v.name for v in tf.trainable_variables()]
        # tvalues = sess.run(tvariables_names)
        # for k, v in zip(tvariables_names, tvalues):
        #     print "Variable: ", k
        # print "Global Variables: "
        # gvariables_names = [v.name for v in tf.global_variables()]
        # gvalues = sess.run(gvariables_names)
        # for k, v in zip(gvariables_names, gvalues):
        #     print "Variable: ", k
        # print "Moving Average Variables: "
        # mvariables_names = [v.name for v in tf.moving_average_variables()]
        # mvalues = sess.run(mvariables_names)
        # for k, v in zip(mvariables_names, mvalues):
        #     print "Variable: ", k

        # Start the queue runners.
        tf.train.start_queue_runners(sess)

        g.finalize()

        # Run training loop
        # run_train_loop(sess, (loss_op, train_op, summary_op), saver)
        run_train_loop(sess, (loss_op, train_op))


def main():
    """
    Creates checkpoint directory to save training progress and records
    training parameters in a json file before initiating the training session.
    """
    if ARGS.train_dir != ARGS.checkpoint:
        if tf.gfile.Exists(ARGS.train_dir):
            tf.gfile.DeleteRecursively(ARGS.train_dir)
        tf.gfile.MakeDirs(ARGS.train_dir)

    # Dump command line arguments to a parameter file,
    # in case the network training resumes at a later time.
    with open(os.path.join(ARGS.train_dir,
                           'deepSpeech_parameters.json'), 'w') as outfile:
        json.dump(vars(ARGS), outfile, sort_keys=True, indent=4)

    args = setenvs(sys.argv)
 
    train()

if __name__ == '__main__':
    main()
