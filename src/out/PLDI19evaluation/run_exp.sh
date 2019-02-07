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
python3 generate_cifar10_data.py --data-dir cifar10_data


echo "Exp: run squeezenet models first"
cd squeezenet
cd pytorch
echo "Note: if you haven't generate onnx model from the PyTorch implementation, do it now by uncommenting the command below."
echo "Note: without the onnx model, you cannot generate Lantern code. You need to generate Lantern code too."
# python3 train.py --generate_onnx ../squeezenetCifar10.onnx
echo "Exp: run PyTorch training with GPU"
python3 train.py --use_gpu=True
echo "Exp: run PyTorch inference with GPU"
python3 train.py --use_gpu=True --inference=True --write_to=result_PyTorch_inference_GPU
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
cp pytorch/result_PyTorch result_PyTorch.txt
cp lantern/result_Lantern result_Lantern.txt
cp tensorflow/result_TensorFlow result_TensorFlow.txt
python3 ../plot.py SqueezeNet result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt


echo "Exp: run ResNet50 models"
cd ../resnet50
cd pytorch
echo "Note: if you haven't generate onnx model from the PyTorch implementation, do it now by uncommenting the command below."
echo "Note: without the onnx model, you cannot generate Lantern code. You need to generate Lantern code too."
# python3 train.py --generate_onnx ../resnet50.onnx
echo "Exp: run PyTorch training with GPU"
python3 train.py --use_gpu=True
echo "Exp: run PyTorch inference with GPU"
python3 train.py --use_gpu=True --inference=True --write_to=result_PyTorch_inference_GPU
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
cp pytorch/result_PyTorch result_PyTorch.txt
cp lantern/result_Lantern result_Lantern.txt
cp tensorflow/result_TensorFlow result_TensorFlow.txt
python3 ../plot.py ResNet50 result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt


echo "Exp: run TreeLSTM models"
cd ../treelstm
echo "Exp: run Lantern training with GPU"
cd lantern
nvcc -g -ccbin gcc-5 -std=c++11 -O3 --expt-extended-lambda -Wno-deprecated-gpu-targets -lstdc++ LanternTraining.cu -o LanternTrainingCu -lcublas -lcudnn
./LanternTrainingCu result_Lantern
echo "Exp: run PyTorch training with GPU"
cd ../pytorch
python3 treeLSTM.py --use_gpu=True
echo "Exp: run TensorFold training with GPU"
cd ../tensorflow
echo "Note: tensorFold only works with tensorflow 1.0. We need to set up python virtual env for it"
echo "Note: if you have not set up the virtual env for tensorfold, uncomment the following lines to set up venv"
# python3 -m venv fold-env
source fold-env/bin/activate
# pip3 install --upgrade pip wheel
# pip3 install --upgrade tensorflow-gpu==1.0.0  # this version of tensorflow works with cuda 8.
# pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
python3 TreeLSTMTensorFlow.py result_TensorFold20
deactivate
cd ../dynet
echo "Exp: run Dynet training (without autobatching) with GPU"
python3 treelstmDynet.py result_DyNetNB --dynet-gpus 1
echo "Exp: run Dynet training (with autobatching) with GPU"
python3 treelstmDynet.py result_DyNetB --dynet-gpus 1 --dynet-autobatch 1
cd ../
cp lantern/result_Lantern result_Lantern.txt
cp pytorch/result_PyTorch result_PyTorch.txt
cp tensorflow/result_TensorFold20 result_TF20.txt
cp dynet/result_DyNetNB result_DyNetNB.txt
cp dynet/result_DyNetB result_DyNetB.txt
python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TF20.txt result_DyNetNB.txt result_DyNetB.txt


echo "Exp: run DeepSpeech2 here"
echo "Exp: run pytorch deepspeech2"
cd ../deepspeech2
cd ds2-pytorch/pytorch
python3 train.py
cd ../../lantern
echo "Exp: run lantern deepspeech2"
nvcc -g -ccbin gcc-5 -std=c++11 -O3 --expt-extended-lambda -Wno-deprecated-gpu-targets -lstdc++ Lantern.cu -o Lantern -lcublas -lcudnn
./Lantern result_Lantern
cd ../
cp ds2-pytorch/pytorch/result_PyTorch result_PyTorch.txt
cp lantern/result_Lantern result_Lantern.txt
python3 ../plot.py DeepSpeech2 result_Lantern.txt result_PyTorch.txt
