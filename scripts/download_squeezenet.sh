#!/bin/bash

DIR=$HOME/onnx_models/squeezenet

if [[ ! -d $DIR ]]
then
	mkdir -p $HOME/tmp
	mkdir -p $HOME/onnx_models
	wget https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz -O $HOME/tmp/squeezenet.tar.gz
	tar xzf $HOME/tmp/squeezenet.tar.gz -C $HOME/onnx_models
	rm -f $HOME/tmp/squeezenet.tar.gz
fi
