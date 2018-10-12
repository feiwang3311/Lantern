import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib

#data_dir = "../senti"

#sample = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
#         0   1      2  3   4     5  6  7     8  9          10          11 12      13 14 15         16                17 18     19 20  21      22      23                24     
#sample2= "(2 (2 (1 No) (2 one)) (1 (1 (2 goes) (2 (1 (2 (2 unindicted) (2 here)) (2 ,)) (2 (2 which) (3 (2 (2 is) (2 probably)) (3 (2 for) (4 (2 the) (4 best))))))) (2 .)))"

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
# we assume the 2G glove original file is at this directory
glove_path = '../PyTorch/data/glove/glove.840B.300d.txt'
# we want to filter it so that we maintain a small subset of the glove embeddings locally
filtered_glove_path = os.path.join('./', 'filtered_glove.txt')
def filter_glove(words):
  ncount = 0
  with codecs.open(glove_path, encoding = 'utf-8') as f:
    with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in words:
          out.write(line + '\n')
          ncount += 1
  print("Lantern: generating filtered_glove, taking %s lines" % ncount)

# if we already have the filtered file locally, then no need to filter again
if not os.path.exists(filtered_glove_path):
  print("Lantern: need to filter the big 2G GOLVE file into a smaller subset file called %s" % filtered_glove_path)
  filter_glove(words)

# we also want to generate a file containing array[double] only, for Lantern to read.
# also this function generates the word_idx, which is a mapping from word to index in the array[double]s
dev_glove_path = os.path.join('./', 'small_glove.txt')
print("Lantern: now generate embedding file called %s and word_idx" % dev_glove_path)
def filter_small_glove(words):
  nread = 0
  nwrote = 0
  word_idx = {}

  # first we need to figure out how many lines we will write
  ncount = 0
  with codecs.open(filtered_glove_path, encoding='utf-8') as f:
    with codecs.open(dev_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        line = line.strip()
        if not line: continue
        temp = line.split(u' ', 1)
        if temp[0] in words:
          ncount += 1
      # add a random row of 300 numbers for unseen words
      ncount += 1

  # then we actually write to the file
  with codecs.open(filtered_glove_path, encoding='utf-8') as f:
    with codecs.open(dev_glove_path, 'w', encoding='utf-8') as out:
      out.write(str(ncount) + '\n') # write the number of entries in this file 
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        temp = line.split(u' ', 1) 
        if temp[0] in words:
          #out.write(temp[0] + ' ')
          out.write(temp[1] + '\n')
          word_idx[temp[0]] = nwrote
          nwrote += 1
      # add a random row of 300 number, for unseen words
      rn = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)  
      for i in range(len(rn)):
        if i == len(rn) - 1: out.write(str(rn[i]) + '\n')
        else: out.write(str(rn[i]) + ' ') 
  print('Lantern: read %s lines, wrote %s' % (nread, nwrote + 1))
  return (nwrote), word_idx
  
# filter Glove file and get word -> index relationship
index_unknown, word_idx = filter_small_glove(words)

# parse samples so that we have tree encoded as arrays
def parseOneSample(data):

  def secondCompleteEnclosing(data, i):
    count = 0
    while (True):
      i += 1
      if data[i].endswith(')'):
        count -= (data[i].count(')') - 1)
        if count == 0:
          return (i+1)
      else: 
        count += 1

  data_raw = data.split()
  data = []
  for i in range(len(data_raw)):
    if data_raw[i].endswith(')'):
      data[-1] = data[-1] + ' ' + data_raw[i]
    else:
      data.append(data_raw[i])

  scores = []
  values = []
  for i in data:
    scores.append(int(i[1:].split()[0]))
    if i.endswith(')'):
      entry = i.split()[-1].split(')')[0]
      if entry in word_idx.keys(): encode = word_idx[entry]
      else: encode = index_unknown
      values.append(encode)
    else:
      values.append(-1)
  
  lch = []
  rch = []
  for i in range(len(data)):
    if data[i].endswith(')'):
      lch.append(-1)
      rch.append(-1)
    else:
      lch.append(i+1)
      rch.append(secondCompleteEnclosing(data, i))
  # return the arrays
  return scores, values, lch, rch

# parse samples in dev.txt file and write array_encode of trees to file
array_tree_path = os.path.join('./', 'array_tree.txt')
def write_array_tree():
  # read target_file, for each line, call parseOneSample to get arrays
  i = 0
  with open(target_file, "r") as f:
    with open(array_tree_path, 'w') as out:
      out.write(str(num_tree) + '\n')
      for line in f:
        arrays = parseOneSample(line)
        i += 1
        out.write(str(len(arrays[0])) + '\n')
        for array in arrays:
          out.write(' '.join(str(item) for item in array) + '\n')
  print("Lantern: wrote %s data entries to %s" % (i, array_tree_path))
write_array_tree()