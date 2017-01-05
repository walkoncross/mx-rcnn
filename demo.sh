#!/usr/bin/env sh

export MXNET_ENGINE_TYPE=NaiveEngine

set -x
python -u predict.py --prefix model/faster-resnet-50 --epoch 0 --img data/det.jpg --gpu 0 --thresh 0.7 --min-size 24 --nms-thresh 0.3 --nest-thresh 0.6
