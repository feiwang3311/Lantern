#!/bin/bash

#scripts/run_resnet50_once.sh 1
#scripts/run_resnet50_once.sh 1
# I run this again, mostly because the GPU seems to have a "wear out effect"
# that is to say, the first run is fast, but the following are not.
# so I woul rather overwrite the results of the first run with more stable runs.
#scripts/run_resnet50_once.sh 2
#scripts/run_resnet50_once.sh 3

python3 plot_stats.py ResNet50 resnet50/results/
