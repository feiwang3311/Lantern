#!/bin/bash

echo "Note: make sure you are using the most updated .cpp file!"

cd evaluationRNN
echo "Note: we can only do -O2 flag now, -O3 has segfault"
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
python min-char-rnn.py result_Numpy.txt
python min-char-rnn-pytorch.py result_PyTorch.txt
python min-char-rnn-tf.py result_TensorFlow.txt
../plot.py vanilla_RNN result_Lantern.txt result_Numpy.txt result_PyTorch.txt result_TensorFlow.txt
cd ..


cd evaluationLSTM
echo "Note: we can only do -O2 flag now, -O3 has segfault"
g++ -std=c++11 -O2 -Wno-pointer-arith Lantern.cpp -o Lantern
./Lantern result_Lantern.txt
python min-char-lstm-pytorch.py result_PyTorch.txt
python min-char-lstm-tf.py result_TensorFlow.txt
../plot.py LSTM result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
cd ..
