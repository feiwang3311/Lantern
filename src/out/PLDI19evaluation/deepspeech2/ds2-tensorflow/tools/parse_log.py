import fileinput as fin

# funcs:
def findValWithFormat(line):
	lines.append(line)
	taken = line.split(" ")
	raw_val = taken[-1]
	val = raw_val.split("/")[-1]
	val = val[0:-2]
	if 'us' in val:
		val = float(val[0:val.find('us')])
		val = val/1000
	else:
		val = float(val[0:val.find('ms')])
	return val

def getCellNum(line):
	cell_num = line[line.find(rnn_cell_string):line.find(rnn_cell_string) + len(rnn_cell_string) + 1]
	return cell_num

def profRNNCell(line, rnncell_prof):
	cell_num = getCellNum(line)
	val = findValWithFormat(line)
	rnncell_prof[cell_num] += val

# variables:
lines = []
module_rnncell = "CustomRNNCell2"
module_grad = 'gradients'
num_rnn_layer = 7
rnn_cell_string = "cell_"
module_rnn = 'rnn'
module_conv1 = 'conv1'
module_conv2 = 'conv2'
module_softmax = 'softmax_linear'
module_ctc = ['ctc_loss', 'CTCLoss']
module_bn = 'bn2'

rnn_cells = [rnn_cell_string+str(i) for i in range(num_rnn_layer)]

rnncell_f_prof = dict.fromkeys(rnn_cells)
rnncell_b_prof = dict.fromkeys(rnn_cells)

# prf estimator:
for el in rnncell_f_prof:
	rnncell_f_prof[el] = 0.0
for el in rnncell_b_prof:
	rnncell_b_prof[el] = 0.0

overall_cost = 0.0

profs ={\
  'rnn_trans_f_prof': 0.0, \
  'rnn_trans_b_prof': 0.0, \
  'rnn_reshape_f_prof': 0.0, \
  'rnn_reshape_b_prof': 0.0, \
  'rnn_ReverseSequence_f_prof': 0.0, \
  'rnn_ReverseSequence_b_prof': 0.0, \
  'conv1_f_prof': 0.0, \
  'conv1_b_prof': 0.0, \
  'bn1_f_prof': 0.0, \
  'bn1_b_prof': 0.0, \
  'relu1_f_prof': 0.0, \
  'relu1_b_prof': 0.0, \
  'conv2_f_prof': 0.0, \
  'conv2_b_prof': 0.0, \
  'bn2_f_prof': 0.0, \
  'bn2_b_prof': 0.0, \
  'relu2_f_prof': 0.0, \
  'relu2_b_prof': 0.0, \
  'softmax_f_prof': 0.0, \
  'softmax_b_prof': 0.0, \
  'ctc_f_prof': 0.0, \
  'ctc_b_prof': 0.0 \
	}


with open('timing_memory.log', 'r') as f:
	for line in f:
		if len(line) > 3:
			if ((line[3] != ' ') or 'Adam/update_' in line) and ('flops' not in line):
				# flops is not considered
				# conv1
				if (module_grad not in line) and (module_conv1 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['conv1_f_prof'] += val
				if (module_grad in line) and (module_conv1 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['conv1_b_prof'] += val

				# BN1
				if (module_grad not in line) and (module_conv1 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn in line):
					val = findValWithFormat(line)
					profs['bn1_f_prof'] += val
				if (module_grad in line) and (module_conv1 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn in line):
					val = findValWithFormat(line)
					profs['bn1_b_prof'] += val

				# Relu1
				if (module_grad not in line) and (module_conv1 in line) and ('Minimum' in line or 'Relu' in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['relu1_f_prof'] += val
				if (module_grad in line) and (module_conv1 in line) and ('Minimum' in line or 'Relu' in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['relu1_b_prof'] += val

				# conv2
				if (module_grad not in line) and (module_conv2 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['conv2_f_prof'] += val
				if (module_grad in line) and (module_conv2 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['conv2_b_prof'] += val

				# BN2
				if (module_grad not in line) and (module_conv2 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn in line):
					val = findValWithFormat(line)
					profs['bn2_f_prof'] += val
				if (module_grad in line) and (module_conv2 in line) and ('Minimum' not in line) and ('Relu' not in line) and (module_bn in line):
					val = findValWithFormat(line)
					profs['bn2_b_prof'] += val

				# Relu2
				if (module_grad not in line) and (module_conv2 in line) and ('Minimum' in line or 'Relu' in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['relu2_f_prof'] += val
				if (module_grad in line) and (module_conv2 in line) and ('Minimum' in line or 'Relu' in line) and (module_bn not in line):
					val = findValWithFormat(line)
					profs['relu2_b_prof'] += val

				#rnn transpose
				if (module_grad not in line) and (module_rnn in line) and ('transpose' in line) and (module_rnncell not in line):
					val = findValWithFormat(line)
					profs['rnn_trans_f_prof'] += val
				if (module_grad in line) and (module_rnn in line) and ('transpose' in line) and (module_rnncell not in line):
					val = findValWithFormat(line)
					profs['rnn_trans_b_prof'] += val

				#rnn reshape
				if (module_grad not in line) and (module_rnn in line) and ('rnn/Reshape' in line) and (module_rnncell not in line):
					val = findValWithFormat(line)
					profs['rnn_reshape_f_prof'] += val
				if (module_grad in line) and (module_rnn in line) and ('rnn/Reshape' in line) and (module_rnncell not in line):
					val = findValWithFormat(line)
					profs['rnn_reshape_b_prof'] += val

				#rnn reshape
				if (module_grad not in line) and (module_rnn in line) and ('ReverseSequence' in line):
					val = findValWithFormat(line)
					profs['rnn_ReverseSequence_f_prof'] += val
				if (module_grad in line) and (module_rnn in line) and ('ReverseSequence' in line):
					val = findValWithFormat(line)
					profs['rnn_ReverseSequence_b_prof'] += val

				# rnn forward profiling by cell
				if (module_grad not in line) and (module_rnncell in line):
					profRNNCell(line, rnncell_f_prof)
				# rnn backward profiling by cell
				if (module_grad in line) and (module_rnncell in line):
					profRNNCell(line, rnncell_b_prof)

				# softmax
				if (module_grad not in line) and (module_softmax in line):
					val = findValWithFormat(line)
					profs['softmax_f_prof'] += val
				if (module_grad in line) and (module_softmax in line):
					val = findValWithFormat(line)
					profs['softmax_b_prof'] += val

				# ctc
				for c in module_ctc:
					if (c in line) and (module_grad not in line):
						val = findValWithFormat(line)
						profs['ctc_f_prof'] += val
					if (c in line) and (module_grad in line):
						val = findValWithFormat(line)
						profs['ctc_b_prof'] +=val


for key, val in dict.iteritems(rnncell_f_prof):
	overall_cost += val
	print "(RNN forward by cell) " + str(key) + ": " + str(val) + "ms"
for key, val in dict.iteritems(rnncell_b_prof):
	overall_cost += val
	print "(RNN backward by cell) " + str(key) + ": " + str(val) + "ms"


# Profiling result
for k in dict.fromkeys(profs):
	overall_cost += profs[k]
	print k + ": " + str(profs[k]) + "ms"

print "overall: " + str(overall_cost) + "ms"


prf_file1 = open('prf1.txt', 'w')
for k in dict.fromkeys(profs):
	prf_file1.write("%s:%f\n" % (k, profs[k]))
prf_file1.close()

# write including modules
prf_file2 = open('prf2.txt', 'w')
for el in lines:
  prf_file2.write("%s\n" % el)
prf_file2.close()














