import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

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
  if label == 'PyTorch':
    return 'r'
  if label == 'TensorFlow':
    return 'y'
  if label == 'TensorFold' or label == 'TensorFold20' or label == 'TF20':
    return 'm'
  if label == 'DyNet' or label == 'DyNetB':
    return 'g'
  if label == 'DyNetNB':
    return 'c'
  else:
    print("NOTE: color not defined for label: %s" % label)

def plot(files, model):
  # save dir
  save_dir = '../save_fig/'
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  datas = {}
  labels = []
  for file1 in files:
    label = getLabelFromFileName(file1)
    datas[label] = get_data(file1)
    labels.append(label)
  print(labels)

  # accumulate data of loss
  losses = []
  for label in labels:
    losses.append(datas[label][2])
  # accumulate data of runtime
  prepareTimes = []
  loopTimes = []
  for label in labels:
    # prepareTimes.append(datas[label][1][0])
    loopTimes.append(float(datas[label][1][1]))
  print(loopTimes)

  # get unit and other description
  unit = datas[labels[0]][0]
  print(unit)
  if (unit == ' 1 epoch'):
    steps = len(losses[0])
    step_desc = "1 epoch"
  else:
    steps = len(losses[0]) - 1
    temp = unit.split()
    step_desc = str(int(temp[0]) * steps) + " " + temp[1] + "s"

  # plot
  fig, ax = plt.subplots()
  N = len(labels)
  ind = np.arange(1, N+1)

  ps = plt.bar(ind, loopTimes, width = 0.55)
  for i in range(N):
    ps[i].set_facecolor(getColor(labels[i]))
  ax.set_xticks(ind)
  ax.set_xticklabels(labels, fontsize = 16)
  ax.tick_params(axis='y', labelsize = 15)
  ax.set_ylim([0, max(loopTimes) * 1.2])
  ax.set_ylabel("seconds", fontsize = 20)
  if step_desc == "1 epoch":
    ax.set_title("{} training time per epoch".format(model), fontsize = 20)
  else:
    ax.set_title("{} training time in {}".format(model, step_desc), fontsize = 20)
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
