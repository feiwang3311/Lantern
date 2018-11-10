import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib

data_dir = "../senti"

def getAllwordsFromOneData(data):
  data = data.split()
  words = set()
  for i in data:
    if i.endswith(')'):
      words.add(i.split(')')[0])
  return (words)

#get all words used in dev.txt and collect the number of trees
target_file = './dev.txt'
#target_file = os.path.join(data_dir, target)
words = set()
num_tree = 0
with open(target_file, 'r') as f:
  for line in f:
    num_tree += 1
    words.update(getAllwordsFromOneData(line))

#filter the Golve file for all words used, so we don't have to keep a big file in memory
glove_path = '../PyTorch/data/glove/glove.840B.300d.txt'
#filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')

# we will save the filted file in here:
dev_glove_path = os.path.join('./', 'small_glove.txt')
def filter_small_glove(words):
  nread = 0
  nwrote = 0
  with codecs.open(glove_path, encoding='utf-8') as f:
    with codecs.open(dev_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in words:
          out.write(line + '\n')
          nwrote += 1
  print('read %s lines, wrote %s' % (nread, nwrote))
# check if the filtered file already exists. if not, run filter_small_glove to generate it
if not os.path.exists(dev_glove_path):
  print("First let's filter the big 2G GLOVE embedding data to a smaller subset that we use")
  filter_small_glove(words)
  print("small glove file successfully generated")