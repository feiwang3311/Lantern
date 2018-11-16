# Author: Lakshmi Krishnan
# Email: lkrishn7@ford.com
# Author: YAO Matrix
# Email: yaoweifeng0301@126.com


"""Builds the deepSpeech network.

Summary of major functions:

  # Compute input feats and labels for training. 
  inputs, labels, seq_len = inputs()

  # Compute inference on the model inputs to make a prediction.
  predictions = inference(inputs)

  # Compute the total loss of the prediction with respect to the labels.
  loss = loss(predictions, labels)

"""


import tensorflow as tf

from helper_routines import _variable_on_cpu
from helper_routines import _variable_with_weight_decay
# from helper_routines import _activation_summary
import custom_ops
import deepSpeech_input

# Global constants describing the speech data set.
NUM_CLASSES = deepSpeech_input.NUM_CLASSES
NUM_PER_EPOCH_FOR_TRAIN = deepSpeech_input.NUM_PER_EPOCH_FOR_TRAIN
NUM_PER_EPOCH_FOR_EVAL = deepSpeech_input.NUM_PER_EPOCH_FOR_EVAL
NUM_PER_EPOCH_FOR_TEST = deepSpeech_input.NUM_PER_EPOCH_FOR_TEST


def get_rnn_seqlen(seq_lens):
    # seq_lens = tf.Print(seq_lens, [seq_lens], "Original seq len: ", 32)
    seq_lens = tf.cast(seq_lens, tf.float64)
    rnn_seq_lens = tf.div(tf.subtract(seq_lens, 10), 2.0)
    rnn_seq_lens = tf.ceil(rnn_seq_lens)
    rnn_seq_lens = tf.div(tf.subtract(rnn_seq_lens, 10), 1.0)
    rnn_seq_lens = tf.ceil(rnn_seq_lens)
    rnn_seq_lens = tf.cast(rnn_seq_lens, tf.int32)

 
    # rnn_seq_lens = tf.Print(rnn_seq_lens, [rnn_seq_lens], "Conved seq len: ", summarize=32)
    # print "rnn_seq_lens shape: ", rnn_seq_lens.get_shape().as_list()
    return rnn_seq_lens


def inputs(eval_data, data_dir, batch_size, use_fp16, shuffle):
    """Construct input for LibriSpeech model evaluation using the Reader ops.

    Args:
      eval_data: 'train', 'test' or 'eval'
      data_dir: folder containing the pre-processed data
      batch_size: int,size of mini-batch
      use_fp16: bool, if True use fp16 else fp32
      shuffle: bool, to shuffle the tfrecords or not. 

    Returns:
      feats: Spetrogram. 3D tensor of [batch_size, T, F] size.
      labels: Labels. 1D tensor of [batch_size] size.
      seq_lens: SeqLens. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    print('Using LibriSpeech Corpus')
    feats, labels, seq_lens = deepSpeech_input.inputs(eval_data=eval_data,
                                                      data_dir=data_dir,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)
    return feats, labels, seq_lens


def inference(sess, feats, seq_lens, params):
    """Build the deepSpeech model.

    Args:
      feats: MFCC features returned from distorted_inputs() or inputs().
      seq_lens: Input sequence length per utterance.
      params: parameters of the model.

    Returns:
      logits.
    """
    # data layout: N, T, F
    # feat_len = feats.get_shape().as_list()[-1]
    # print "feat shape: ", feats.get_shape().as_list()

    #########################
    #  convolutional layers
    #########################
    with tf.variable_scope('conv1') as scope:
        ## N, T, F
        feats = tf.expand_dims(feats, axis=3)

        ## N, T, F, 1
        # convolution
        kernel = _variable_with_weight_decay('weights',
                                             shape=[11, 41, 1, params.num_filters],
                                             wd_value=None,
                                             use_fp16=params.use_fp16)
        conv = tf.nn.conv2d(feats, kernel,
                            [1, 2, 2, 1],
                            padding='VALID')
        # biases = _variable_on_cpu('biases', [params.num_filters],
        #                           tf.constant_initializer(-0.05),
        #                          params.use_fp16)
        # bias = tf.nn.bias_add(conv, biases)

        ## N, T, F, 32
        # batch normalization
        bn = custom_ops.batch_norm(conv)

        # clipped ReLU
        conv1 = custom_ops.relux(bn, capping=20)
#        _activation_summary(conv1)

    with tf.variable_scope('conv2') as scope:
        ## N, T, F, 32
        # convolution
        kernel = _variable_with_weight_decay('weights',
                                             shape=[11, 21, params.num_filters, params.num_filters],
                                             wd_value=None,
                                             use_fp16=params.use_fp16) 
        conv = tf.nn.conv2d(conv1,
                            kernel,
                            [1, 1, 2, 1],
                            padding='VALID')
        # biases = _variable_on_cpu('biases',
        #                           [params.num_filters],
        #                           tf.constant_initializer(-0.05),
        #                           params.use_fp16)
        # bias = tf.nn.bias_add(conv, biases)

        ## N, T, F, 32
        # batch normalization
        bn = custom_ops.batch_norm(conv)

        # clipped ReLU
        conv2 = custom_ops.relux(bn, capping=20)
#        _activation_summary(conv2)

    ######################
    # recurrent layers
    ######################
    # Reshape conv output to fit rnn input: N, T, F * C
    fdim = conv2.get_shape().dims
    feat_dim = fdim[2].value * fdim[3].value
    rnn_input = tf.reshape(conv2, [params.batch_size, -1, feat_dim])

    # Permute into time major order for rnn: T, N, F * C
    rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
    # T is unknown. N = 32, F * C = 672
    # rnn_input = tf.unstack(rnn_input)

    # fw_cell = custom_ops.CustomRNNCell2(params.num_hidden)
    # fw_cell_list = [fw_cell] * params.num_rnn_layers

    # bw_cell = custom_ops.CustomRNNCell2(params.num_hidden)
    # bw_cell_list = [bw_cell] * params.num_rnn_layers

    fw_cells = [tf.nn.rnn_cell.BasicRNNCell(params.num_hidden) for i in range(params.num_rnn_layers)]
    bw_cells = [tf.nn.rnn_cell.BasicRNNCell(params.num_hidden) for i in range(params.num_rnn_layers)]
    conved_seq_lens = get_rnn_seqlen(seq_lens)

    rnn_outputs = custom_ops.stacked_brnn(fw_cells, bw_cells, rnn_input, params.batch_size, conved_seq_lens)
    print(rnn_outputs.get_shape())
#    _activation_summary(rnn_outputs)

    # Linear layer(WX + b) - softmax is applied by CTC cost function.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [NUM_CLASSES, params.num_hidden * 2],
                                              wd_value=None,
                                              use_fp16=params.use_fp16)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0),
                                  params.use_fp16)
        logit_inputs = tf.reshape(rnn_outputs, [-1, params.num_hidden * 2])
        logits = tf.add(tf.matmul(logit_inputs, weights, transpose_a=False, transpose_b=True),
                        biases, name=scope.name)
        logits = tf.matmul(logit_inputs, weights, transpose_a=False, transpose_b=True)
        logits = tf.reshape(logits, [-1, params.batch_size, NUM_CLASSES])
        # _activation_summary(logits)
    # print(logits.get_shape())
    return logits


def loss(logits, labels, seq_lens):
    """Compute mean CTC Loss.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference(). layout: T N F
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
      seq_lens: Length of each utterance for ctc cost computation.

    Returns:
      Loss tensor of type float.
    """
    # dense_labels = tf.sparse_tensor_to_dense(labels)
    # dense_labels = tf.Print(dense_labels, [dense_labels], "labels: ")

    ## scheme 1
    # logits_shape = tf.shape(logits)
    # logits_shape = tf.Print(logits_shape, [logits_shape], "logits shape: ")
    # max_seq_len = logits_shape[0]
    # batch_size = logits_shape[1]
    # max_seq_len = tf.Print(max_seq_len, [max_seq_len], "max seq len: ")
    # conved_seq_lens = tf.fill([batch_size], max_seq_len)
    # conved_seq_lens = tf.Print(conved_seq_lens, [conved_seq_lens], "conved seq len: ", summarize=32)

    ## scheme 2
    conved_seq_lens = get_rnn_seqlen(seq_lens)

#    conved_seq_lens = tf.Print(conved_seq_lens, [conved_seq_lens], "conved seq len: ", summarize=32)

    # Calculate the average ctc loss across the batch.
    ctc_loss = tf.nn.ctc_loss(labels=labels, inputs=tf.cast(logits, tf.float32),
                              sequence_length=conved_seq_lens, 
                              preprocess_collapse_repeated=False,
                              ctc_merge_repeated=True,
                              time_major=True,
                              ignore_longer_outputs_than_inputs=True)
#    ctc_loss = tf.Print(ctc_loss, [ctc_loss], "CTC loss: ", summarize=32)
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
#    ctc_loss_mean = tf.Print(ctc_loss_mean, [ctc_loss_mean], "mean CTC loss: ")
    # tf.add_to_collection('losses', ctc_loss_mean)

    # The total loss is defined as the cross entropy loss plus all
    # of the weight decay terms (L2 loss).
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return ctc_loss_mean


def _add_loss_summaries(total_loss):
    """Add summaries for losses in deepSpeech model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for each_loss in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.scalar_summary(each_loss.op.name + ' (raw)', each_loss)
        tf.scalar_summary(each_loss.op.name, loss_averages.average(each_loss))

    return loss_averages_op
