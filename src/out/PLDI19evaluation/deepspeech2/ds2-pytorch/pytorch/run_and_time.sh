# Script to train and time DeepSpeech 2 implementation

RANDOM_SEED=1
# TARGET_ACC=23
TARGET_ACC=0

#python train.py --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
CUDA_VISIBLE_DEVICES=3 python train.py  --model_path models/deepspeech_t$RANDOM_SEED.pth.tar --seed $RANDOM_SEED --acc $TARGET_ACC
