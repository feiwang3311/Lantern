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
echo "Exp: run PyTorch interence with CPU"
python3 train.py --inference=True --write_to=result_PyTorch_inference_CPU
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
cp lantern/result_Lantern result_Lantern.txt
cp pytorch/result_PyTorch result_PyTorch.txt
cp tensorflow/result_TensorFold20 result_TF20.txt
cp dynet/result_DyNetNB result_DyNetNB.txt
cp dynet/result_DyNetB result_DyNetB.txt
python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TF20.txt result_DyNetNB.txt result_DyNetB.txt

exit 1

export OPENBLAS_NUM_THREADS=1

source python3-env/bin/activate
cd evaluationRNN
echo "Note: Let's run vanilla RNN experiment first"
echo "RUN: run Lantern"
g++ -std=c++11 -O3 Lantern.cpp -o Lantern -Wno-pointer-arith -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
numactl -C 0 ./Lantern result_Lantern.txt
echo "RUN: run PyTorch"
numactl -C 0 python3 min-char-rnn-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
numactl -C 0 python3 min-char-rnn-tf.py result_TensorFlow.txt
echo "RUN: plotting"
python3 ../plot.py vanilla_RNN result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: vanilla RNN experiment successful"
cd ..

cd evaluationLSTM
echo "Note: Let's run LSTM experiment now"
echo "RUN: run Lantern"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
numactl -C 0 ./Lantern result_Lantern.txt
echo "RUN: run PyTorch"
numactl -C 0 python3 min-char-lstm-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
numactl -C 0 python3 min-char-lstm-tf.py result_TensorFlow.txt
echo "RUN: plotting"
python3 ../plot.py LSTM result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: LSTM experiment successful"
cd ..

cd evaluationCNN
cd PyTorch
echo "Note: Let's run CNN in PyTorch first, which also helps downloading the training data"
echo "Download data and also extract data for Lantern to use"
python3 download_data.py
python3 extract_data.py
echo "RUN: PyTorch CNN with batch size 100 and learning rate 0.05"
numactl -C 0 python3 PyTorch.py
echo "Result: PyTorch CNN run successfully"
cd ..
cd TensorFlow
echo "Note: now Let's run TensorFlow CNN. Need to use another install with MKL support"
deactivate
echo "RUN: TensorFlow CNN with batch size 100 and learning rate 0.05"
numactl -C 0 python3 TensorFlow.py
echo "Result: TensorFlow CNN run successfully"
cd ..
cd Lantern
echo "Note: Let's run Lantern now with batch size 100"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
echo "RUN: Lantern CNN"
numactl -C 0 ./Lantern result_Lantern.txt
echo "Result: Lantern CNN successful"
cd ..
source ../python3-env/bin/activate
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch100.txt result_PyTorch.txt
cp TensorFlow/result_TensorFlow100.txt result_TensorFlow.txt
python3 ../plot.py CNN result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: run CNN experiment successful"
cd ..

cd evaluationTreeLSTM
echo "Note: Let's run TreeLSTM experiment now"
echo "Now let's run Lantern"
cd Lantern
echo "RUN: run Lantern"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
OPENBLAS_NUM_THREADS=1 numactl -C 0 ./Lantern result_Lantern.txt
echo "Result: run sentiment in Lantern is successful"
cd ..
echo "Now let's run Dynet"
cd Dynet
echo "RUN: run dynet without autobatching"
numactl -C 0 python3 treelstmDynet.py result_DyNetNB.txt
echo "RUN: run dynet with autobatching"
numactl -C 0 python3 treelstmDynet.py result_DyNetB.txt --dynet-autobatch 1
echo "Result: run sentiment in Dynet is successful"
cd ..
echo "Now let's run PyTorch"
cd PyTorch
numactl -C 0 python3 treeLSTM.py
cd ..
cd TensorFold
echo "RUN: run TensorFold"
python3 preprocess_data.py

echo "Note: now Let's run TensorFlow Fold. Need to use another install with TensorFlow 1.0 and TensorFold install"
deactivate

# python3 -m venv tensorfold-dev
# pip3 install tensorflow-gpu==1.0.0
# pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
# pip3 install matplotlib

source tensorfold-dev/bin/activate
numactl -C 0 python3 TreeLSTMTensorFlow.py result_TensorFold20.txt
echo "Result: run sentiment in TensorFold is successful"
cd ..
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch.txt result_PyTorch.txt
cp TensorFold/result_TensorFold20.txt result_TensorFold20.txt
cp Dynet/result_DyNetNB.txt result_DyNetNB.txt
cp Dynet/result_DyNetB.txt result_DyNetB.txt
python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TensorFold20.txt result_DyNetNB.txt result_DyNetB.txt # result_TensorFold1.txt
echo "RESULT: run TreeLSTM experiment successful"
cd ..
deactivate
