#!/bin/bash
echo "Note: make sure you are using the most updated .cpp file!"

cd evaluationRNN
echo "Note: Let's run vanilla RNN experiment first"
echo "RUN: run Lantern"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
./Lantern result_Lantern.txt
echo "RUN: run PyTorch"
python3 min-char-rnn-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
python3 min-char-rnn-tf.py result_TensorFlow.txt
echo "RUN: plotting"
python3 ../plot.py vanilla_RNN result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: vanilla RNN experiment successful"
cd ..

cd evaluationLSTM
echo "Note: Let's run LSTM experiment now"
echo "RUN: run Lantern"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
./Lantern result_Lantern.txt
echo "RUN: run PyTorch"
python3 min-char-lstm-pytorch.py result_PyTorch.txt
echo "RUN: run TensorFlow"
python3 min-char-lstm-tf.py result_TensorFlow.txt
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
python3 PyTorch.py
#echo "RUN: PyTorch CNN with batch size 1 and learning rate 0.0005"
#python PyTorch.py --batch-size 1 --lr 0.0005
echo "Result: PyTorch CNN run successfully"
cd ..
cd TensorFlow
echo "Note: now Let's run TensorFlow CNN"
echo "RUN: TensorFlow CNN with batch size 100 and learning rate 0.05"
python3 TensorFlow.py
#echo "RUN: TensorFlow CNN with batch size 1 and learning rate 0.0005"
#python TensorFlow.py --batch-size 1 --lr 0.0005
echo "Result: TensorFlow CNN run successfully"
cd ..
cd Lantern
echo "Note: Let's run Lantern now with batch size 100"
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
echo "RUN: Lantern CNN"
./Lantern result_Lantern.txt
echo "Result: Lantern CNN successful"
cd ..
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch100.txt result_PyTorch.txt
#cp PyTorch/result_PyTorch1.txt result_PyTorch1.txt
cp TensorFlow/result_TensorFlow100.txt result_TensorFlow.txt
#cp TensorFlow/result_TensorFlow1.txt result_TF1.txt
python3 ../plot.py CNN result_Lantern.txt result_PyTorch.txt result_TensorFlow.txt
echo "RESULT: run CNN experiment successful"
cd ..

cd evaluationTreeLSTM
echo "Note: Let's run TreeLSTM experiment now"
echo "Now let's run Lantern"
cd Lantern
echo "RUN: run Lantern"
# python3 preprocess_data.py
g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern -I /opt/OpenBLAS/include -L /opt/OpenBLAS/lib -lopenblas
./Lantern result_Lantern.txt
echo "Result: run sentiment in Lantern is successful"
cd ..
echo "Now let's run Dynet"
cd Dynet
echo "RUN: run dynet without autobatching"
python3 treelstmDynet.py result_DynetNB.txt --dynet-mem 2048
echo "RUN: run dynet with autobatching"
python3 treelstmDynet.py result_DynetB.txt --dynet-autobatch 1 --dynet-mem 2048
echo "Result: run sentiment in Dynet is successful"
cd ..
echo "Now let's run PyTorch"
cd PyTorch
python3 treeLSTM.py
cd ..
cd TensorFold
echo "RUN: run TensorFold"
python3 preprocess_data.py
echo "Note: we assume the system has python-pip python-dev python-virtualenv"
echo "Note: if not, you can install it by uncomment the commands below, but you need root access"
#sudo apt install python-pip python-dev python-virtualenv
echo "Note: we assume the system has ./virEnv setup with tensorflow1.0.0 and fold"
echo "Note: if not, uncomment the commands below to set it up"
#virtualenv -p python3 virEnv
source ./virEnv/bin/activate
#sudo pip3 install tensorflow==1.0.0
#pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
python3 TreeLSTMTensorFlow.py result_TensorFold20.txt
# python TreeLSTMTensorFlow.py result_TensorFold1.txt 1
deactivate
echo "Result: run sentiment in TensorFold is successful"
cd ..
echo "RUN: copy the result files and do plotting"
cp Lantern/result_Lantern.txt result_Lantern.txt
cp PyTorch/result_PyTorch.txt result_PyTorch.txt
cp TensorFold/result_TensorFold20.txt result_TensorFold20.txt
# cp TensorFold/result_TensorFold1.txt result_TensorFold1.txt
cp Dynet/result_DynetNB.txt result_DynetNB.txt
cp Dynet/result_DynetB.txt result_DynetB.txt
python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TensorFold20.txt result_DynetNB.txt result_DynetB.txt # result_TensorFold1.txt
echo "RESULT: run TreeLSTM experiment successful"
cd ..



# cd evaluationTreeLSTM
# echo "Note: Let's run TreeLSTM experiment now"
# cd PyTorch
# echo "RUN: run PyTorch, which will make sure that we have necessary data for Lantern and TensorFold as well"
# echo "Note: adapted from https://github.com/ttpro1995/TreeLSTMSentiment"
# echo "Note: we assume the requirements as indicated in the website are already met (tqdm, etc). Check the website if not sure"
# if [ ! -d "data" ]; then
#   echo "Note: we need to run ./fetch_and_preprcess.sh first, which downloads data and library of size about 2GB"
#   ./fetch_and_preprocess.sh
#   python3 sentiment.py # this extra run is simply to pre-process data
# fi
# python3 sentiment.py
# echo "Result: run sentiment in PyTorch is successful"
# cd ..
# cd TensorFold
# echo "RUN: run TensorFold"
# python3 preprocess_data.py
# echo "Note: we assume the system has python-pip python-dev python-virtualenv"
# echo "Note: if not, you can install it by uncomment the commands below, but you need root access"
# #sudo apt install python-pip python-dev python-virtualenv
# echo "Note: we assume the system has ./virEnv setup with tensorflow1.0.0 and fold"
# echo "Note: if not, uncomment the commands below to set it up"
# #virtualenv -p python3 virEnv
# source ./virEnv/bin/activate
# #pip3 install tensorflow==1.0.0
# #pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
# python3 TreeLSTMTensorFlow.py result_TensorFold20.txt
# # python TreeLSTMTensorFlow.py result_TensorFold1.txt 1
# deactivate
# echo "Result: run sentiment in TensorFold is successful"
# cd ..
# cd Lantern
# echo "RUN: run Lantern"
# python3 preprocess_data.py
# g++ -std=c++11 -O3 -Wno-pointer-arith Lantern.cpp -o Lantern
# ./Lantern result_Lantern.txt
# echo "Result: run sentiment in Lantern is successful"
# cd ..
# cd Dynet
# echo "RUN: run dynet"
# python3 treelstmDynet.py --dynet-autobatch 1 --dynet-mem 2048
# echo "Result: run sentiment in Dynet is successful"
# cd ..
# echo "RUN: copy the result files and do plotting"
# cp Lantern/result_Lantern.txt result_Lantern.txt
# cp PyTorch/result_PyTorch.txt result_PyTorch.txt
# cp TensorFold/result_TensorFold20.txt result_TensorFold20.txt
# # cp TensorFold/result_TensorFold1.txt result_TensorFold1.txt
# cp Dynet/result_Dynet.txt result_Dynet.txt
# python3 ../plot.py TreeLSTM result_Lantern.txt result_PyTorch.txt result_TensorFold20.txt result_Dynet.txt # result_TensorFold1.txt
# echo "RESULT: run TreeLSTM experiment successful"
# cd ..
