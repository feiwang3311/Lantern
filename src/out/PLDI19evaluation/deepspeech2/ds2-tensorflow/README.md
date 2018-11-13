# TensorFlow implementation of DeepSpeech2
End-to-end speech recognition using TensorFlow

This repository contains TensorFlow code for an end-to-end speech recognition engine by implementing Baidu's DeepSpeech2 model on IA architectures. This work was based on the code developed by Ford[https://github.com/fordDeepDSP/deepSpeech] and many changes have been conducted to fin our solution.

This software is released under a BSD license. The license to this software does not apply to TensorFlow, which is available under the Apache 2.0 license, or the third party pre-requisites listed below, which are available under their own respective licenses.

Pre-requisites
-------------
* TensorFlow - version: 1.1.0, 1.2.0
* Python     - version: 2.7
* python-levenshtein - to compute Character-Error-Rate
* python_speech_features - to generate mfcc features
* PySoundFile - to read FLAC files
* scipy - helper functions for windowing
* tqdm - for displaying a progress bar

Getting started
------------------
*Step 1: Install all dependencies.*

```shell
$ yum install libsndfile
$ pip install python-Levenshtein
$ pip install python_speech_features
$ pip install PySoundFile
$ pip install scipy
$ pip install tqdm

# Install TensorFlow 1.2.0:
$ pip install 'tensorflow==1.2.0'

# [GPU ONLY] Update ~/.bashrc to reflect path for CUDA.
1. Add these lines to the ~/.bashrc:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
2. Install TF GPU package
$ pip install --upgrade 'tensorflow-gpu==1.2.0'

```
*Step 2: Clone this git repo.*
```shell
$ git clone https://github.com/yao-matrix/deepSpeech2.git
$ cd deepSpeech
```

Preprocessing the data
----------------------
*Step 1: Download and unpack the LibriSpeech data*

Inside the github repo that you have cloned run:
```shell
$ mkdir -p data/librispeech
$ cd data/librispeech
$ wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
$ wget http://www.openslr.org/resources/12/dev-clean.tar.gz
$ wget http://www.openslr.org/resources/12/test-clean.tar.gz
$ mkdir audio
$ cd audio
$ tar xvzf ../train-clean-100.tar.gz LibriSpeech/train-clean-100 --strip-components=1
$ tar xvzf ../dev-clean.tar.gz LibriSpeech/dev-clean  --strip-components=1
$ tar xvzf ../test-clean.tar.gz LibriSpeech/test-clean  --strip-components=1
# delete audios which are too short
$ rm -rf LibriSpeech/train-clean-100/1578/6379/1578-6379-0029.flac
$ rm -rf LibriSpeech/train-clean-100/460/172359/460-172359-0090.flac
```
*Step 2: Run this command to preprocess the audio and generate TFRecord files.*

The computed mfcc features will be stored within TFRecords files inside data/librispeech/processed/
```shell
$ cd ./src
$ python preprocess_LibriSpeech.py
```

Training a model w/ dummy data
----------------
```shell
$ cd ./src
$ vim ./train.sh
# let dummy=1 in train.sh
$ ./train.sh
```

Training a model w/ real data
----------------
```shell
# To continue training from a saved checkpoint file
$ cd ./src
$ vim ./train.sh
# let dummy=0 in train.sh
$ ./train.sh
```
The script train.sh contains commands to train on utterances in sorted order for the first epoch and then to resume training on shuffled utterances.
Note that during the first epoch, the cost will increase and it will take longer to train on later steps because the utterances are presented in sorted order to the network.

Monitoring training
--------------------
Since the training data is fed through a shuffled queue, to check validation loss a separate graph needs to be set up in a different session. This graph is fed with the valildation data to compute predictions. The deepSpeech_test.py script initializes the graph from a previously saved checkpoint file and computes the CER on the eval_data every 5 minutes by default. It saves the computed CER values in the models/librispeech/eval folder. By calling tensorboard with logdir set to models/librispeech, it is possible to monitor validation CER and training loss during training.
```shell
$ cd ./src
$ ./validation.sh
$ tensorboard --logdir PATH_TO_SUMMARY
```
Testing a model
----------------
```shell
$ cd ./src
$ ./test.sh
```

# Thanks
Thanks to Aswathy for helping refine the README
