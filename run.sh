#!/usr/bin/env sh

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#export MXNET_ENGINE_TYPE=NaiveEngine

set -x
python -u train_end2end.py --gpus 2,3 --year 2007 --image_set trainval --num_epoch 10 --kv_store=device --factor-step 20000 \
        --lr 0.001
