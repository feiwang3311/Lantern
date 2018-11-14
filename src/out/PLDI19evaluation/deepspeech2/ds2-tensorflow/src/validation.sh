#!/bin/bash
# This script trains a deepspeech model in tensorflow with sorta-grad.
# usage ./train.sh  or  ./train.sh dummy


clear
cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
export PYTHONPATH=${cur_path}:/home/matrix/inteltf/:$PYTHONPATH
# echo $PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

# activate Intel Python
# source /opt/intel/intelpython2/bin/activate

# environment variables
unset TF_CPP_MIN_VLOG_LEVEL
# export TF_CPP_MIN_VLOG_LEVEL=1

# clear
echo "-----------------------------------"
echo "Start validation"

nchw=True    # True or False
engine="mkl" # tf, mkl, cudnn_rnn, mkldnn_rnn


config_check_one=`test "${nchw}" = "False" && test "${engine}"x = "tf"x -o "${engine}"x = "cudnn_rnn"x && echo 'OK'`
# echo "check one: "$config_check_one
config_check_two=`test "${nchw}" = "True" && test "${engine}"x == "mkl"x -o "${engine}"x = "mkldnn_rnn"x && echo 'OK'`
# echo "check two: "$config_check_two
check=`test ${config_check_one}x = "OK"x -o ${config_check_two}x = "OK"x && echo 'OK'`
# echo "check: "$check

if [[ ${check}x != "OK"x ]];then
    echo "unsupported configuration conbimation"
    exit -1
fi

python deepSpeech_test.py --eval_data 'val' --nchw ${nchw} --engine ${engine}

echo "Done"

# deactivate Intel Python
# source /opt/intel/intelpython2/bin/deactivate

