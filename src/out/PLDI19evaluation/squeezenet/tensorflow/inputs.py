from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np

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
		x = [i.astype(np.float32).reshape(3, 32, 32) for i in x]
		y = self.labels[self.current_idx: self.current_idx + self.batch_size]
		y = np.asarray(y, dtype=np.int32)
		self.current_idx += self.batch_size
		return x, y
