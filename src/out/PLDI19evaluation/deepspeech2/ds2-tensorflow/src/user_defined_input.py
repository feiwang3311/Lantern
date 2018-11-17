from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
import struct

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

class Batch(object):

  def __init__(self, filename):
    self.dict = unpickle(filename)
    self.numBatches = self.dict[b'numBatches']
    self.batchSize = self.dict[b'batchSize']
    self.batchedData = self.dict[b'batchedData']
    self.current_batch = 0

    def batchWithRawLength(self):
      (_, maxlen, inputs, input_percentages, target_sizes, targets) = self.batchedData[self.current_batch]
      self.current_batch += 1
      if self.current_batch >= self.numBatches:
        self.current_batch = 0
      return inputs, targets, input_percentages, maxlen, target_sizes

  def batch(self):
    (_, _, inputs, input_percentages, target_sizes, targets) = self.batchedData[self.current_batch]
    self.current_batch += 1
    if self.current_batch >= self.numBatches:
      self.current_batch = 0
    return inputs, targets, input_percentages, target_sizes

  def batch_with_metainfo(self):
      (freq_size, maxlen, inputs, input_percentages, target_sizes, targets) = self.batchedData[self.current_batch]
      self.current_batch += 1
      if self.current_batch >= self.numBatches:
        self.current_batch = 0
      return freq_size, maxlen, inputs, targets, input_percentages, target_sizes

  def write_to_bin(self, input_file, target_file):
    with open(input_file, 'wb') as f:
      with open(target_file, 'wb') as g:
        x, y = self.batch()
        for by in x.reshape(-1).tolist():
          f.write(struct.pack('@f', by))
        for by in y.reshape(-1).tolist():
          g.write(struct.pack('@i', int(by)))

if __name__ == '__main__':
  batch = Batch('../../cifar10_data/cifar-10-batches-py/data_batch_1', 64)
  batch.write_to_bin('../cifar10_data/cifar-10-batches-bin/small_batch_x.bin', '../cifar10_data/cifar-10-batches-bin/small_batch_y.bin')
