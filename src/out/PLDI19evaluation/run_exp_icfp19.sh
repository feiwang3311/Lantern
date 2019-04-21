#!/bin/bash

echo "if have not yet run preproces, uncomment it"
# scripts/preprocess.sh

echo "now run the experiments"
scripts/run_deepspeech2.sh
scripts/run_squeezenet.sh
scripts/run_resnet50.sh
scripts/run_treelstm.sh
