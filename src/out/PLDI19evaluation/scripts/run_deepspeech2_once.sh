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

echo "Exp: run DeepSpeech2 here"
echo "Exp: run pytorch deepspeech2"
cd deepspeech2
cd ds2-pytorch/pytorch
python3 train.py

echo "Exp: run lantern deepspeech2"
cd ../../lantern
nvcc -g -ccbin gcc-5 -std=c++11 -O3 --expt-extended-lambda -Wno-deprecated-gpu-targets -lstdc++ Lantern.cu -o Lantern -lcublas -lcudnn
./Lantern result_Lantern
cd ../

mkdir -p results/
cp ds2-pytorch/pytorch/result_PyTorch results/result_PyTorch_$1.txt
cp lantern/result_Lantern results/result_Lantern_$1.txt
# python3 ../plot.py DeepSpeech2 result_Lantern.txt result_PyTorch.txt
