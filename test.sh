#!/usr/bin/env sh

set -x
python -u test.py --has_rpn --prefix model/faster-rcnn --epoch 5 --end2end

