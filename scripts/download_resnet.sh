DIR1=$HOME/onnx_models/resnet

if [[ ! -d $DIR1 ]]
then
	mkdir -p $HOME/tmp
	mkdir -p $HOME/onnx_models
	wget https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz -O $HOME/tmp/resnet.tar.gz
	tar xzf	$HOME/tmp/resnet.tar.gz -C $HOME/onnx_models
	rm -f $HOME/tmp/resnet.tar.gz
fi
