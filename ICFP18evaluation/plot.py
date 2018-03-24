#! /usr/bin/python

def get_data(filename):
  #print("retrieving data from " + filename)
  loss_save = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.rstrip()
      if line.startswith("unit:"):
        unit = line.split(':')[1]
      elif line.startswith("run time:"):
        runtime = line.split(':')[1].split()
        assert(len(runtime) == 2)
      elif len(line) > 0:
        loss_save.append(float(line))
  return [unit, runtime, loss_save]
  
def getLabelFromFileName(filename):
  return filename.split('.')[0].split('_')[1]

def getColor(label):
  if label == 'Numpy':
    return 'g'
  if label == 'Lantern':
    return 'b'
  if label == 'PyTorch' or label == 'PyTorch1':
    return 'r'
  if label == 'PyTorch20' or label == 'PyTorch100':
    return 'c'
  if label == 'TensorFlow' or label == 'TensorFlow1' or label == 'TF' or label == 'TF1':
    return 'y'
  if label == 'TensorFlow20' or label == 'TensorFlow100' or label == 'TF20' or label == 'TF100':
    return 'm'
  if label == 'TensorFold' or label == 'TensorFold1':
    return 'y'
  if label == 'TensorFold20' or label == 'TensorFold100':
    return 'm'
  else:
    print("NOTE: color not defined for label: %s" % label)

def plot(files, model):
  # save dir 
  save_dir = '../save_fig/'
  import os
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  datas = {}
  labels = []
  for file1 in files:
    label = getLabelFromFileName(file1) 
    datas[label] = get_data(file1)
    labels.append(label)

  # now plot
  import numpy as np
  import matplotlib.pyplot as plt
  import pylab
  
  # accumulate data of loss
  losses = []
  for label in labels:
    losses.append(datas[label][2])
  # accumulate data of runtime
  prepareTimes = []
  loopTimes = []
  for label in labels:
    prepareTimes.append(datas[label][1][0])
    loopTimes.append(datas[label][1][1])
  
  # get unit and other description
  unit = datas[labels[0]][0]
  print(unit)
  if (unit == ' 1 epoch'): 
    steps = len(losses[0])
  else: 
    steps = len(losses[0]) - 1
  temp = unit.split()
  step_desc = str(int(temp[0]) * steps) + " " + temp[1] + "s"
  

  # plot 
  plt.figure(1, figsize=(18, 6))
  plt.subplot(121)
  for i in range(len(labels)): 
    plt.plot(losses[i], getColor(labels[i]) + '-', linewidth=2.0, label = labels[i])
  plt.legend()  
  plt.title("training loss over " + step_desc)
  plt.xlabel('number of ' + unit + 's')
  plt.ylabel('loss')
  #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
  #plt.axis([40, 160, 0, 0.03])
  #plt.grid(True)
  plt.subplot(122)
  width = 0.5
  space = 0.5
  start = space + 0.25
  bars = []
  for i in range(len(labels)):
    plt.bar([start], [loopTimes[i]], width, color=getColor(labels[i]))
    plt.bar([start], [prepareTimes[i]], width, bottom=[loopTimes[i]], color = 'k')
    start = start + width + space
  import matplotlib.patches as mpatches
  black_patch = mpatches.Patch(color='black', label='prepare time')
  mix_patch = mpatches.Patch()
  plt.legend(handles=[black_patch], bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
  plt.title("prepare time and iteration time of " + step_desc)
  plt.ylabel("seconds")
  plt.xticks((np.arange(len(labels)) + 1), labels)
  #plt.show()
  pylab.savefig(save_dir + model + '.png')


if __name__ == "__main__":
  import sys
  #print(sys.argv)
  model = sys.argv[1]  
  n_files = len(sys.argv) - 2
  files = []
  for i in range(n_files):
    files.append(sys.argv[i+2])
  plot(files, model)