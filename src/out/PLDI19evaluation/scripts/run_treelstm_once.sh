#!/bin/bash
echo "Note: make sure you are using the most updated .cpp file!"
echo "Note: we assume the system has python-pip python-dev python-virtualenv"
echo "Note: the script must be run in PLDIevaluation directory"
echo "Note: the evaluation is done with a single GPU"
echo "Note: we assume that a proper python virtual environment has be installed"
export CUDA_VISIBLE_DEVICES=3

echo "Note: we are using the newest tensorflow pytorch installation in /scratch-ml00/wang603/"
echo "Note: Maybe source the conda environment? Not sure"
source /scratch-ml00/wang603/conda3/bin/activate

echo "Exp: run TreeLSTM models"
cd treelstm
echo "Exp: run Lantern training with GPU"
cd lantern
nvcc -g -std=c++11 -O3 --expt-extended-lambda -Wno-deprecated-gpu-targets -lstdc++ LanternTraining.cu -o LanternTrainingCu -lcublas -lcudnn
./LanternTrainingCu result_Lantern

echo "Exp: run PyTorch training with GPU"
cd ../pytorch
python3 treeLSTM.py --use_gpu=True

echo "Exp: run TensorFold training with GPU"
cd ../tensorflow
echo "Note: tensorFold only works with tensorflow 1.0. We need to set up python virtual env for it"
echo "Note: if you have not set up the virtual env for tensorfold, uncomment the following lines to set up venv"
#python3 -m venv fold-env
source fold-env/bin/activate
#pip3 install --upgrade pip wheel
#pip3 install --upgrade tensorflow-gpu==1.0.0  # this version of tensorflow works with cuda 8.
#pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
python3 TreeLSTMTensorFlow.py result_TensorFold20
deactivate

cd ../dynet
echo "Exp: run Dynet training (without autobatching) with GPU"
python3 treelstmDynet.py result_DyNetNB --dynet-gpus 1

echo "Exp: run Dynet training (with autobatching) with GPU"
python3 treelstmDynet.py result_DyNetB --dynet-gpus 1 --dynet-autobatch 1

cd ../
mkdir -p results
cp lantern/result_Lantern results/result_Lantern_$1.txt
cp pytorch/result_PyTorch results/result_PyTorch_$1.txt
cp tensorflow/result_TensorFold20 results/result_TF20_$1.txt
cp dynet/result_DyNetNB results/result_DyNetNB_$1.txt
cp dynet/result_DyNetB results/result_DyNetB_$1.txt
# python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TF20.txt result_DyNetNB.txt result_DyNetB.txt
