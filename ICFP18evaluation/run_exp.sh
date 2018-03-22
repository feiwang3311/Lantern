#!/bin/bash
echo "Note: make sure you are using the most updated .cpp file!"


cd evaluationTreeLSTM
echo "Note: Let's run TreeLSTM experiment now"
cd PyTorch
echo "RUN: run PyTorch, which will make sure that we have necessary data for Lantern and TensorFold as well"
echo "Note: adapted from https://github.com/ttpro1995/TreeLSTMSentiment"
echo "Note: we assume the requirements as indicated in the website are already met (tqdm, etc). Check the website if not sure"
if [ ! -d "data" ]; then
  echo "Note: we need to run ./fetch_and_preprcess.sh first, which downloads data and library of size about 2GB"
  ./fetch_and_preprocess.sh
  python sentiment.py # this extra run is simply to pre-process data
fi
python sentiment.py
echo "Result: run sentiment in PyTorch is successful"
cd ..
cd TensorFold
echo "RUN: run TensorFold"
python preprocess_data.py
echo "Note: we assume the system has python-pip python-dev python-virtualenv"
echo "Note: if not, you can install it by uncomment the commands below, but you need root access"
#sudo apt install python-pip python-dev python-virtualenv
echo "Note: we assume the system has ./virEnv setup with tensorflow1.0.0 and fold"
echo "Note: if not, uncomment the commands below to set it up"
#virtualenv -p python3 virEnv
source ./virEnv/bin/activate
#pip3 install tensorflow==1.0.0
#pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
python TreeLSTMTensorFlow.py result_TensorFold20.txt
python TreeLSTMTensorFlow.py result_TensorFold1.txt 1
deactivate
echo "Result: run sentiment in TensorFold is successful"
cd ..
cd Lantern
echo "RUN: run Lantern"
echo "Note: we can only do -O2 flag now, -O3 has segfault"
python preprocess_data.py
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
echo "Result: run sentiment in Lantern is successful"
cd ..
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch.txt result_PyTorch.txt
cp TensorFold/result_TensorFold20.txt result_TensorFold20.txt
cp TensorFold/result_TensorFold1.txt result_TensorFold1.txt
../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TensorFold20.txt result_TensorFold1.txt
echo "RESULT: run TreeLSTM experiment successful"
cd ..




cd evaluationRNN
echo "Note: Let's run vanilla RNN experiment first"
echo "RUN: run Lantern"
echo "Note: we can only do -O2 flag now, -O3 has segfault"
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
echo "RUN: run Numpy"
python min-char-rnn.py result_Numpy.txt
echo "RUN: run PyTorch"
python min-char-rnn-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
python min-char-rnn-tf.py result_TensorFlow.txt
echo "RUN: plotting"
../plot.py vanilla_RNN result_Lantern.txt result_Numpy.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: vanilla RNN experiment successful"
cd ..


cd evaluationLSTM
echo "Note: Let's run LSTM experiment now"
echo "RUN: run Lantern"
echo "Note: we can only do -O2 flag now, -O3 has segfault"
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
echo "RUN: run PyTorch"
python min-char-lstm-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
python min-char-lstm-tf.py result_TensorFlow.txt
echo "RUN: plotting"
../plot.py LSTM result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: LSTM experiment successful"
cd ..


cd evaluationTreeLSTM
echo "Note: Let's run TreeLSTM experiment now"
cd PyTorch
echo "RUN: run PyTorch, which will make sure that we have necessary data for Lantern and TensorFold as well"
echo "Note: adapted from https://github.com/ttpro1995/TreeLSTMSentiment"
echo "Note: we assume the requirements as indicated in the website are already met (tqdm, etc). Check the website if not sure"
if [ ! -d "data" ]; then
  echo "Note: we need to run ./fetch_and_preprcess.sh first, which downloads data and library of size about 2GB"
  ./fetch_and_preprocess.sh
  python sentiment.py # this extra run is simply to pre-process data
fi
python sentiment.py
echo "Result: run sentiment in PyTorch is successful"
cd ..
cd TensorFold
echo "RUN: run TensorFold"
python preprocess_data.py
echo "Note: we assume the system has python-pip python-dev python-virtualenv"
echo "Note: if not, you can install it by uncomment the commands below, but you need root access"
#sudo apt install python-pip python-dev python-virtualenv
echo "Note: we assume the system has ./virEnv setup with tensorflow1.0.0 and fold"
echo "Note: if not, uncomment the commands below to set it up"
#virtualenv -p python3 virEnv
source ./virEnv/bin/activate
#pip3 install tensorflow==1.0.0
#pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
python TreeLSTMTensorFlow.py result_TensorFold20.txt
python TreeLSTMTensorFlow.py result_TensorFold1.txt 1
deactivate
echo "Result: run sentiment in TensorFold is successful"
cd ..
cd Lantern
echo "RUN: run Lantern"
echo "Note: we can only do -O2 flag now, -O3 has segfault"
python preprocess_data.py
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
echo "Result: run sentiment in Lantern is successful"
cd ..
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch.txt result_PyTorch.txt
cp TensorFold/result_TensorFold20.txt result_TensorFold20.txt
cp TensorFold/result_TensorFold1.txt result_TensorFold1.txt
../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TensorFold20.txt result_TensorFold1.txt
echo "RESULT: run TreeLSTM experiment successful"
cd ..