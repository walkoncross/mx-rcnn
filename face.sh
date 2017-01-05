#!/usr/bin/env sh

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#export MXNET_ENGINE_TYPE=NaiveEngine

set -x
python -u train.py --gpus 4,5,6,7 --num_epoch 20 --dataset-root /home/work/data/Face/WIDER \
       --pretrained model/faster-resnet-50 --load-epoch 13 --resume \
       --factor-step 50000 --frequent 20 --lr 0.0001 
