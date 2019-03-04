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

echo "Note: Maybe downloading cifar10_data"
# python3 generate_cifar10_data.py --data-dir cifar10_data

echo "Exp: run ResNet50 models"
cd resnet50
cd pytorch
echo "Note: if you haven't generate onnx model from the PyTorch implementation, do it now by uncommenting the command below."
echo "Note: without the onnx model, you cannot generate Lantern code. You need to generate Lantern code too."
# python3 train.py --generate_onnx ../resnet50.onnx

echo "Exp: run PyTorch training with GPU"
python3 train.py --use_gpu=True
# echo "Exp: run PyTorch inference with GPU"
# python3 train.py --use_gpu=True --inference=True --write_to=result_PyTorch_inference_GPU
# echo "Exp: run PyTorch interence with CPU"
# python3 train.py --inference=True --write_to=result_PyTorch_inference_CPU

cd ../lantern
echo "Exp: run Lantern training with GPU"
nvcc -g -ccbin gcc-5 -std=c++11 -O3 --expt-extended-lambda -Wno-deprecated-gpu-targets -lstdc++ LanternOnnxTraining.cu -o LanternOnnxTrainingCu -lcublas -lcudnn
./LanternOnnxTrainingCu	result_Lantern

cd ../tensorflow
echo "Exp: run TensorFlow training with GPU"
python3 train.py
echo "Plot: plot squeezenet result"

cd ..
mkdir -p results
cp pytorch/result_PyTorch results/result_PyTorch_$1.txt
cp lantern/result_Lantern results/result_Lantern_$1.txt
cp tensorflow/result_TensorFlow results/result_TensorFlow_$1.txt
# python3 ../plot.py ResNet50 result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
