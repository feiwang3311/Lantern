import argparse
import errno
import json
import os
import time

import sys

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

### Import Data Utils ###
sys.path.append('../')

from model import DeepSpeech, supported_rnns
import user_defined_input

import params

###########################################################
# Comand line arguments, handled by params except seed    #
###########################################################
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')

parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')

parser.add_argument('--acc', default=23.0, type=float, help='Target WER')

parser.add_argument('--start_epoch', default=0, type=int, help='Number of epochs at which to start from')
parser.add_argument('--write_to', default='result_PyTorch', type=str, help='where to save the performance')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')

def to_np(x):
    return x.data.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
      print("ERROR: GRU does not currently support activations other than tanh")
      sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
      print("ERROR: We should be using ReLU RNNs")
      sys.exit()

    print("=======================================================")
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(params.epochs), torch.Tensor(params.epochs), torch.Tensor(params.epochs)
    best_wer = None
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size = params.hidden_size,
                       nb_layers       = params.hidden_layers,
                       labels          = labels,
                       rnn_type        = supported_rnns[rnn_type],
                       audio_conf      = None,
                       bidirectional   = True,
                       rnn_activation  = params.rnn_act_type,
                       bias            = params.bias)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=params.lr,
                                momentum=params.momentum, nesterov=False,
                                weight_decay = params.l2)
    cuda = torch.device('cuda')
    criterion = torch.nn.CTCLoss(reduction='none').to(cuda)


    avg_loss = 0
    start_epoch = 0
    start_iter = 0
    avg_training_loss = 0
    if params.cuda:
        model.cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ctc_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    filename = "/scratch/wu636/Lantern/src/out/PLDI19evaluation/deepspeech2/ds2-pytorch/data/test/deepspeech_train.pickle"
    batchedData = user_defined_input.Batch(filename)

    def train_one_epoch(epoch):
        avg_loss = 0
        for i in range(batchedData.numBatches):
            end = time.time()
            inputs, targets, input_percentages, target_sizes = batchedData.batch()

            # making all inputs Tensor
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            input_percentages = torch.from_numpy(input_percentages)
            target_sizes = torch.from_numpy(target_sizes)
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)

            if params.cuda:
                inputs = inputs.cuda()

            # measure forward pass time
            forward_start_time = time.time()
            out = model(inputs)
            # out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            # measure ctc loss computing time
            ctc_start_time = time.time()
            out = out.log_softmax(2)  #.detach().requires_grad_()
            # print(sizes.shape)
            # print(out.shape)
            loss = criterion(out, targets, sizes, target_sizes)
            ctc_time.update(time.time() - ctc_start_time)

            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss_sum.data.item()

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            forward_time.update(time.time() - forward_start_time)

            # measure backward pass time
            backward_start_time = time.time()
            # compute gradient
            optimizer.zero_grad()
            loss_sum.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), params.max_norm)
            # SGD step
            optimizer.step()

            if params.cuda:
                torch.cuda.synchronize()

            backward_time.update(time.time() - backward_start_time)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (i % 20 == 0):
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                  'CTC Time {ctc_time.val:.3f} ({ctc_time.avg:.3f})\t'
                  'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), batchedData.numBatches, batch_time=batch_time,
                data_time=data_time, forward_time=forward_time, ctc_time=ctc_time,
                backward_time=backward_time, loss=losses))

            del loss
            del out

        avg_loss /= batchedData.numBatches #  len(train_loader)

        print('Training Summary Epoch: [{0}]\t'
            'Average Loss {loss:.3f}\t'

            .format(epoch + 1, loss=avg_loss, ))

        return avg_loss

    model.train()
    loss_save = []
    time_save = []
    for epoch in range(start_epoch, args.epochs):
        startTime = time.time()
        loss_save.append(train_one_epoch(epoch))
        endTime = time.time()
        time_save.append(endTime - startTime)
        print("epoch {} used {} seconds".format(epoch, endTime - startTime))

    time_save.sort()
    median_time = time_save[int(args.epochs / 2)]
    with open(args.write_to, "w") as f:
        f.write("unit: " + "1 epoch\n")
        for loss in loss_save:
            f.write("{}\n".format(loss))
        f.write("run time: " + str(0.0) + " " + str(median_time) + "\n")

if __name__ == '__main__':
    main()
