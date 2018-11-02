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

	def __init__(self, filename, batch_size):
		self.dict = unpickle(filename)
		self.data = self.dict[b'data']
		self.labels = self.dict[b'labels']
		self.batch_size = batch_size
		self.current_idx = 0
		self.total_size = len(self.labels)

	def batch(self):
		if (self.current_idx + self.batch_size >= self.total_size):
			self.current_idx = 0
		x = self.data[self.current_idx: self.current_idx + self.batch_size]
		x = [i.astype(np.float32).reshape(3, 32, 32) / 255 for i in x]
		y = self.labels[self.current_idx: self.current_idx + self.batch_size]
		y = np.asarray(y, dtype=np.int64)
		self.current_idx += self.batch_size
		return np.asarray(x), y

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
	batch.write_to_bin('../../cifar10_data/cifar-10-batches-bin/small_batch_x.bin', '../cifar10_data/cifar-10-batches-bin/small_batch_y.bin')

