import fileinput as fin
from collections import OrderedDict
import argparse
import json
from operator import itemgetter
import os
DEBUG = True
# Global variables
time_cond = ['ts', 'dur']
global_vas = OrderedDict([ \
                          ('markTF', ''), \
                          ('gobal_beg_time', 0), \
                          ('output_folder', 'Output'),\
                          ])
layers_names = OrderedDict([ \
          ('conv1_f', 'conv1_forward'), \
          ('conv1_b', 'conv1_backward'), \
          ('bn1_f', 'bn1_forward'), \
          ('bn1_b', 'bn1_backward'), \
          ('relu1_f', 'relu1_forward'), \
          ('relu1_b', 'relu1_backward'), \
          ('conv2_f', 'conv2_forward'), \
          ('conv2_b', 'conv2_backward'), \
          ('bn2_f', 'bn2_forward'), \
          ('bn2_b', 'bn2_backward'), \
          ('relu2_f', 'relu2_forward'), \
          ('relu2_b', 'relu2_backward'), \
          ('rnn_trans_f', 'rnn_transpose_forward'), \
          ('rnn_trans_b', 'rnn_transpose_backward'), \
          ('rnn_reshape_f', 'rnn_reshape_forward'), \
          ('rnn_reshape_b', 'rnn_reshape_backward'), \
          ('rnn_Revs_f', 'rnn_ReverseSequence_forward'), \
          ('rnn_Revs_b', 'rnn_ReverseSequence_backward'), \
          ('rnn_f_0', 'rnn_forward_cell_0'), \
          ('rnn_b_0', 'rnn_backward_cell_0'), \
          ('rnn_f_1', 'rnn_forward_cell_1'), \
          ('rnn_b_1', 'rnn_backward_cell_1'), \
          ('rnn_f_2', 'rnn_forward_cell_2'), \
          ('rnn_b_2', 'rnn_backward_cell_2'), \
          ('rnn_f_3', 'rnn_forward_cell_3'), \
          ('rnn_b_3', 'rnn_backward_cell_3'), \
          ('rnn_f_4', 'rnn_forward_cell_4'), \
          ('rnn_b_4', 'rnn_backward_cell_4'), \
          ('rnn_f_5', 'rnn_forward_cell_5'), \
          ('rnn_b_5', 'rnn_backward_cell_5'), \
          ('rnn_f_6', 'rnn_forward_cell_6'), \
          ('rnn_b_6', 'rnn_backward_cell_6'), \
          ('softmax_f', 'softmax_forward'), \
          ('softmax_b', 'softmax_backward'), \
          ('ctc_f', 'ctc_forward'), \
          ('ctc_b', 'ctc_backward'), \
          ('ema', 'ExponentialMovingAverage') \
          ])
# Define ops per layer
layers_ops = OrderedDict()
for key, val in layers_names.iteritems():
    layers_ops[val] = []
# Define modules  
module_grad = 'gradients'
module_rnncell = "CustomRNNCell2/"
rnn_cell_string = "cell_"
module_rnn = 'rnn'
module_conv1 = 'conv1'
module_conv2 = 'conv2'
module_bn = '/bn'
module_softmax = 'softmax_linear'
module_Exponential = 'ExponentialMovingAverage/AssignMovingAvg_'
module_trans = 'transpose'
module_resp ='rnn/Reshape'
module_rvseq = 'ReverseSequence'
module_relu = ['Minimum', 'Relu']
module_ctc = ['ctc_loss', 'CTCLoss']
# Define cond for branch instructions
# opt_exc = ['biases/ApplyAdam', 'weights/ApplyAdam', 'Conv2D_grad/Shape']
opt_inc = OrderedDict()
opt_exc = OrderedDict()
opt_any = OrderedDict()
opt_inc[layers_names['conv1_f']] = [module_conv1]
opt_exc[layers_names['conv1_f']] = [module_grad, module_bn, 'Minimum', 'Relu']
opt_inc[layers_names['conv1_b']] = [module_conv1, module_grad]
opt_exc[layers_names['conv1_b']] = [module_bn, 'Minimum', 'Relu']
opt_inc[layers_names['bn1_f']] = [module_conv1, module_bn]
opt_exc[layers_names['bn1_f']] = [module_grad, 'Minimum', 'Relu']
opt_inc[layers_names['bn1_b']] = [module_conv1, module_bn, module_grad]
opt_exc[layers_names['bn1_b']] = ['Minimum', 'Relu']
opt_inc[layers_names['relu1_f']] = [module_conv1]
opt_exc[layers_names['relu1_f']] = [module_grad, module_bn]
opt_inc[layers_names['relu1_b']] = [module_conv1, module_grad]
opt_exc[layers_names['relu1_b']] = [module_bn]
# conv2
opt_inc[layers_names['conv2_f']] = [module_conv2]
opt_exc[layers_names['conv2_f']] = [module_grad, module_bn, 'Minimum', 'Relu']
opt_inc[layers_names['conv2_b']] = [module_conv2, module_grad]
opt_exc[layers_names['conv2_b']] = [module_bn, 'Minimum', 'Relu']
opt_inc[layers_names['bn2_f']] = [module_conv2, module_bn]
opt_exc[layers_names['bn2_f']] = [module_grad, 'Minimum', 'Relu']
opt_inc[layers_names['bn2_b']] = [module_conv2, module_bn, module_grad]
opt_exc[layers_names['bn2_b']] = ['Minimum', 'Relu']
opt_inc[layers_names['relu2_f']] = [module_conv2]
opt_exc[layers_names['relu2_f']] = [module_grad, module_bn]
opt_inc[layers_names['relu2_b']] = [module_conv2, module_grad]
opt_exc[layers_names['relu2_b']] = [module_bn]
# full connection layer
opt_inc[layers_names['softmax_f']] = [module_softmax]
opt_exc[layers_names['softmax_f']] = [module_grad]
opt_inc[layers_names['softmax_b']] = [module_softmax, module_grad]
opt_exc[layers_names['softmax_b']] = []
opt_inc[layers_names['ctc_f']] = []
opt_exc[layers_names['ctc_f']] = [module_grad]
opt_inc[layers_names['ctc_b']] = [module_grad]
opt_exc[layers_names['ctc_b']] = []

opt_any[layers_names['relu1_f']] = module_relu
opt_any[layers_names['relu1_b']] = module_relu
opt_any[layers_names['relu2_f']] = module_relu
opt_any[layers_names['relu2_b']] = module_relu
opt_any[layers_names['ctc_f']] = module_ctc
opt_any[layers_names['ctc_b']] = module_ctc
debug_exc = []
for i in range(7):
    if i>=0:
        debug_exc.append("rnn_backward_cell_"+str(i))
        debug_exc.append("rnn_forward_cell_"+str(i))
# rnn
cell = "cell_"
i1 = 0
i2 = 0
for key, val in layers_names.iteritems():
    if 'rnn_f' in key:
        opt_inc[val] = [module_rnncell, cell+str(i1)]
        opt_exc[val] = [module_grad]
        i1 += 1
    elif 'rnn_b' in key:
        opt_inc[val] = [module_rnncell, cell+str(i2), module_grad]
        opt_exc[val] = []
        i2 += 1
# others
opt_inc[layers_names['rnn_trans_f']] = [module_rnn, module_trans]
opt_exc[layers_names['rnn_trans_f']] = [module_grad, module_rnncell]
opt_inc[layers_names['rnn_trans_b']] = [module_rnn, module_trans, module_grad]
opt_exc[layers_names['rnn_trans_b']] = [module_rnncell]
opt_inc[layers_names['rnn_reshape_f']] = [module_rnn, module_resp]
opt_exc[layers_names['rnn_reshape_f']] = [module_grad, module_rnncell]
opt_inc[layers_names['rnn_reshape_b']] = [module_rnn, module_resp, module_grad]
opt_exc[layers_names['rnn_reshape_b']] = [module_rnncell]
opt_inc[layers_names['rnn_Revs_f']] = [module_rnn, module_rvseq]
opt_exc[layers_names['rnn_Revs_f']] = [module_grad]
opt_inc[layers_names['rnn_Revs_b']] = [module_rnn, module_rvseq, module_grad]
opt_exc[layers_names['rnn_Revs_b']] = []
opt_inc[layers_names['ema']] = [module_Exponential]
opt_exc[layers_names['ema']] = [module_rnn]

# Functions and classes
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str,
                        default = 'ITF_profiling_32.json')
    args = parser.parse_args()
    return args

def updateGlobalVas(args):
    if 'ITF' in args.input:
        global_vas['markTF'] = 'ITF_'
    elif 'PTF' in args.input:
        global_vas['markTF'] = 'PTF_'
    else:
        global_vas['markTF'] = 'MTF_'
    if not os.path.exists(os.path.join(os.getcwd(), global_vas['output_folder'])):
        os.makedirs(os.path.join(os.getcwd(), global_vas['output_folder']))
    if not os.path.exists(os.path.join(os.getcwd(), global_vas['output_folder'] + '/inc_ops')):
        os.makedirs(os.path.join(os.getcwd(), global_vas['output_folder'] + '/inc_ops'))
    if not os.path.exists(os.path.join(os.getcwd(), global_vas['output_folder'] + '/rnn_gaps')):
        os.makedirs(os.path.join(os.getcwd(), global_vas['output_folder'] + '/rnn_gaps'))


class RawPeriodList:
    def __init__(self):
        self.beg_time = 0.0
        self.end_time = 0.0
        self.beg_op = ''
        self.end_op = ''
        self.periodList = []
        self.period = OrderedDict([\
                                      ('layer_name', ''), \
                                      ('period', 0.0), \
                                      ('beg_time', 0.0), \
                                      ('end_time', 0.0), \
                                      ('beg_op', ''),\
                                      ('end_op', '')])
    def initPeriod(self):
        self.beg_time = 0.0
        self.end_time = 0.0      
    def createPeriod(self, op):
        if self.beg_time == 0 and self.end_time == 0:
                self.beg_op = op['name']
                self.end_op = op['name']
                self.beg_time = op['beg']
                self.end_time = op['end']
        elif self.beg_time > op['beg']:
                self.beg_op = op['name']
                self.beg_time = op['beg']
        elif self.end_time < op['end']:
                self.end_op = op['name']
                self.end_time = op['end']
    def append2List(self, layer_name):
        self.period['layer_name'] = layer_name
        self.period['period'] = self.end_time - self.beg_time
        self.period['beg_time'] = self.beg_time
        self.period['end_time'] = self.end_time
        self.period['beg_op'] = self.beg_op
        self.period['end_op'] = self.end_op
        self.periodList.append(self.period.copy())
    def getRawPeriodList(self):
        return self.periodList
    def printPeriods(self):
        file_name = global_vas['output_folder'] + '/'+ global_vas['markTF'] + 'periods.csv'
        fp = open(file_name, 'w')
        fp.write("layer_name, period, beg_time, beg_op, end_time, end_op\n")
        for period in self.periodList:
            fp.write("%s, %g, %g, %s, %g, %s\n" % (period['layer_name'], period['period'],\
                           period['beg_time'], period['beg_op'],\
                           period['end_time'], period['end_op']))
        fp.close()
        
class TimeStampsList:
    stampsList = []
    stamps = OrderedDict()
    stamp = OrderedDict()

    def __init__(self, layers_ops):
        self.layers = layers_ops
        self.rawPeriodList = RawPeriodList()
        self.stamps = OrderedDict([\
                                      ('layer_name', ''),\
                                      ('stamps', [])])
        self.stamp = OrderedDict([\
                                      ('time', 0.0),\
                                      ('op_name', ''),\
                                      ('pos', '')])
    def initStamps(self):
        self.stamps['stamps'] = []
    def createTimeStamp(self, op):
        self.stamp['time'] = op['beg']
        self.stamp['op_name'] = op['name']
        self.stamp['pos'] = 'beg'
        stamp_beg = self.stamp.copy()
        self.stamp['time'] = op['end']
        self.stamp['op_name'] = op['name']
        self.stamp['pos'] = 'end'
        stamp_end = self.stamp.copy()
        return stamp_beg, stamp_end
    def createStampsList(self):
        for layer_name, ops in self.layers.iteritems():
            self.initStamps()
            self.stamps['layer_name'] = layer_name
            self.rawPeriodList.initPeriod()
            for op in ops:
                stamp_beg, stamp_end = self.createTimeStamp(op)
                self.stamps['stamps'].append(stamp_beg)
                self.stamps['stamps'].append(stamp_end)
                self.rawPeriodList.createPeriod(op)
            self.stamps['stamps'].sort(key = lambda x: x['time'])
            # append the stamps of the layer into list
            self.stampsList.append(self.stamps.copy())
            self.rawPeriodList.append2List(layer_name)         
    def getStampsList(self):
        return self.stampsList
    def getRawPeriodList(self):
        return self.rawPeriodList.getRawPeriodList()
    def printStamps(self):
        file_name = global_vas['output_folder'] + '/'+ global_vas['markTF'] +'stamps.csv'
        fp = open(file_name, 'w')
        for stamps in self.stampsList:
            fp.write("%s\n" % stamps['layer_name'])
            fp.write("time, op_name, pos\n")
            for stamp in stamps['stamps']:
                fp.write("%g, %s, %s\n" % (stamp['time'], stamp['op_name'], stamp['pos']))
        fp.close()
    def printPeriods(self):
        self.rawPeriodList.printPeriods()

class TimeInfo:
    def __init__(self, layers_ops, threshold):
        self.threshold = threshold
        self.rawPeriodList = RawPeriodList()
        self.timeStampsList = TimeStampsList(layers_ops)
        self.layerExeTimeList = LayerExeTimeList(layers_ops)
        self.layers = layers_ops

    def createInfo(self):
        self.timeStampsList.createStampsList()
        stampsList =  self.timeStampsList.getStampsList()
        self.layerExeTimeList.createExeTimeList(stampsList, self.threshold)

        self.timeStampsList.printStamps()
        self.timeStampsList.printPeriods()
        self.layerExeTimeList.printExeTimes()
        self.layerExeTimeList.printGapsList()

class LayerExeTimeList:
    def __init__(self, layers):
        self.interGapsList = InterGapsList()
        self.layers = layers
        self.exeTimeList = []
        self.exeTime = OrderedDict([\
                                      ('layer_name', ''),\
                                      ('wall_time', 0.0),\
                                      ('wall_time_thres', 0.0)])
    def checkInsideOp(self, mid, layer_name):
        ops = self.layers[layer_name]
        for op in ops:
            if mid > op['beg'] and mid < op['end']:
                return True
        return False
    def createExeTimeList(self, stampsList, threshold=0.0):
        # Compute the wall time for layers
        for item in stampsList:
            # For one layer
            layer_name = item['layer_name']
            self.exeTime['layer_name'] = layer_name
            stamps = item['stamps']
            wallTime = 0.0
            wallTime_thres = 0.0
            self.interGapsList.initGaps()
            for i in range(len(stamps)):
                if i == 0:
                    continue
                prev = stamps[i-1]['time']
                current = stamps[i]['time']
                period = current - prev
                mid = prev + period/2
                if self.checkInsideOp(mid, layer_name):
                    wallTime += period
                    wallTime_thres += period
                elif period < threshold:
                    wallTime_thres += period
                    self.interGapsList.append2Gaps(layer_name, period, stamps[i-1], stamps[i])
            self.exeTime['wall_time'] = wallTime
            self.exeTime['wall_time_thres'] = wallTime_thres
            self.exeTimeList.append(self.exeTime.copy())
            self.interGapsList.append2GapsList()
            print ("ExeTime of %s has been computed" % (layer_name))
            
    def getExeTimeList(self):
        return self.exeTimeList
    def printExeTimes(self):
        file_name = global_vas['output_folder'] + '/'+ global_vas['markTF'] +'exeTime.csv'
        fp = open(file_name, 'w')
        fp.write("layer_name, wall_time, wall_time_thres\n")
        for exeTime in self.exeTimeList:
            fp.write("%s, %g, %g\n" % (exeTime['layer_name'], exeTime['wall_time'], exeTime['wall_time_thres']))
        fp.close()
    def printGapsList(self):
        self.interGapsList.printGapsList()
        
class InterGapsList:
    def __init__(self):
        self.gapsList = []
        self.gaps = OrderedDict([\
                                 ('layer_name', ''),\
                                 ('gaps', [])])
        self.gap = OrderedDict([\
                                ('period', 0.0),\
                                ('beg_time', 0.0),\
                                ('end_time', 0.0),\
                                ('beg_op', 0.0),\
                                ('end_op', 0.0)])
    def initGaps(self):
        self.gaps['gaps'] = []
    def append2Gaps(self, layer_name, period, prev, current):
        self.gap['period'] = period
        self.gap['beg_time'] = prev['time']
        self.gap['beg_op'] = prev['op_name']
        self.gap['end_time'] = current['time']
        self.gap['end_op'] = current['op_name']
        self.gaps['layer_name'] = layer_name
        self.gaps['gaps'].append(self.gap.copy())
    def append2GapsList(self):
        self.gapsList.append(self.gaps.copy())
    def printGapsList(self):
        file_name = global_vas['output_folder'] + '/'+ global_vas['markTF'] +'gaps.csv'
        fp = open(file_name, 'w')
        for gaps in self.gapsList:
            if 'cell' in gaps['layer_name']:
                file_name = global_vas['output_folder'] + '/rnn_gaps/'+ global_vas['markTF'] + gaps['layer_name'] +'gaps.csv'
                fp_rnn = open(file_name, 'w')
                for gap in gaps['gaps']:
                    fp_rnn.write("%g, %g, %s, %g, %s\n" % \
                             (gap['period'],gap['beg_time'],gap['beg_op'],gap['end_time'],gap['end_op']))
                fp_rnn.close()
            else:
                fp.write("%s\n" % (gaps['layer_name']))
                for gap in gaps['gaps']:
                    fp.write("%g, %g, %s, %g, %s\n" % \
                             (gap['period'],gap['beg_time'],gap['beg_op'],gap['end_time'],gap['end_op']))
        fp.close()

class Operator:
    opInfo = OrderedDict()
    def __init__(self):
        self.opInfo = OrderedDict([\
                                   ('name', ''),\
                                   ('beg', 0.0),\
                                   ('end', 0.0) ])
             
    def createOpInfo(self, name, beg, end):
        self.opInfo['name'] = name
        self.opInfo['beg'] = float(beg)/1000.0
        self.opInfo['end'] = float(end)/1000.0
           
    def getOpInfo(self):
        return self.opInfo

    def insert(self, alist, input, input_name):
        beg = input['ts'] - global_vas['gobal_beg_time']
        end = beg + input['dur']
        self.createOpInfo(input_name, beg, end)
        alist.append(self.getOpInfo().copy())

class OpsList:   
    gmark = False
    op = Operator()
    layers = OrderedDict()
    
    def __init__(self, layers_ops):
        self.layers = layers_ops

    def recordBegTime(self, item):
        if self.gmark == False and all(c in item.iterkeys() for c in time_cond):
            self.gmark = True
            global_vas['gobal_beg_time'] = item['ts']
            
    def groupByLayer(self, input, cmd):
        for key, layer_name in layers_names.iteritems():
            if 'relu' in key or 'ctc' in key:
                if all(c in cmd for c in opt_inc[layer_name]) and all(c not in cmd for c in opt_exc[layer_name]) \
                    and any(c in cmd for c in opt_any[layer_name]):
                    self.op.insert(self.layers[layer_name], input, cmd)
                    break
            else:
                if all(c in cmd for c in opt_inc[layer_name]) and all(c not in cmd for c in opt_exc[layer_name]):
                    self.op.insert(self.layers[layer_name], input, cmd)
                    break
                       
    def append2List(self, input):    
        input_name = str(input["name"])
        self.recordBegTime(input)   
        if "args" in input.iterkeys() and all(c in input.iterkeys() for c in time_cond):
            input_name = str(input["args"]["name"])
            self.groupByLayer(input, input_name)           
        elif "args" not in input.iterkeys() and all(c in input.iterkeys() for c in time_cond):
            self.groupByLayer(input, input_name) 

    def printList(self):
        for key, val in self.layers.iteritems():
            file_name = global_vas['output_folder'] + '/inc_ops/'+ global_vas['markTF'] +'include_ops_'+key+'.log'
            fp = open(file_name, 'w')
            for v in val:
                fp.write("%s:\n" % (v))
            fp.close()

# Read Json file
args = parse_args()
updateGlobalVas(args)
json_data=open(args.input).read()
jdata = json.loads(json_data)

# Create layer's OPs list
opsList = OpsList(layers_ops)
for item in jdata["traceEvents"]:  
    opsList.append2List(item)

print 'opsList has been created'
opsList.printList()

threshold = 50000.0
timeInfo = TimeInfo(layers_ops, threshold)
timeInfo.createInfo()

















