import argparse
import logging
import os

import sys
# sys.path.insert(0, '/home/work/wuwei/project/dmlc/mxnet/python')
sys.path.insert(0, '/home/work/wuwei/project/mxnet/python')
import mxnet as mx

from rcnn.callback import Speedometer
from rcnn.config import config
from rcnn.loader import AnchorLoader
from rcnn.metric import AccuracyMetric, LogLossMetric, SmoothL1LossMetric
from rcnn.module import MutableModule
from rcnn.symbol import get_faster_rcnn
from utils.load_data import load_gt_roidb
from utils.load_model import load_param

def end2end_train(image_set, test_image_set, year, root_path, devkit_path, pretrained, epoch, prefix,
                  ctx, begin_epoch, num_epoch, frequent, kv_store, work_load_list=None, resume=False):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config.TRAIN.BG_THRESH_LO = 0.0

    logging.info('########## TRAIN FASTER-RCNN WITH APPROXIMATE JOINT END2END #############')
    config.TRAIN.HAS_RPN = True
    config.TRAIN.END2END = 1
    # config.TRAIN.IMS_PER_BATCH = 1
    # config.TRAIN.RCNN_BATCH_SIZE = 256
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
    config.TRAIN.BATCH_SIZE = 256
    # load symbol
    sym = get_faster_rcnn(is_train=True)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # load training data
    voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path, flip=False)
    train_data = AnchorLoader(feat_sym, roidb, batch_size=config.TRAIN.IMS_PER_BATCH, shuffle=True, mode='train',
                              ctx=ctx, work_load_list=work_load_list)
    # infer max shape
    max_data_shape = [('data', (config.TRAIN.IMS_PER_BATCH, 3, 1000, 1000))]
    max_data_shape_dict = {k: v for k, v in max_data_shape}
    _, feat_shape, _ = feat_sym.infer_shape(**max_data_shape_dict)
    from rcnn.minibatch import assign_anchor
    import numpy as np
    label = assign_anchor(feat_shape[0], np.zeros((config.TRAIN.BATCH_SIZE, 5)), [[1000, 1000, 1.0]])
    max_label_shape = [('label', label['label'].shape),
                       ('bbox_target', label['bbox_target'].shape),
                       ('bbox_inside_weight', label['bbox_inside_weight'].shape),
                       ('bbox_outside_weight', label['bbox_outside_weight'].shape),
                       ('gt_boxes', label['gt_boxes'].shape)]
    print 'providing maximum shape', max_data_shape, max_label_shape

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True)
    del args['fc8_weight']
    del args['fc8_bias']

    # initialize params
    if not resume:
        input_shapes = {k: v for k, v in train_data.provide_data + train_data.provide_label}
        arg_shape, _, _ = sym.infer_shape(**input_shapes)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        # import pdb; pdb.set_trace()

        args['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        args['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        args['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        args['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        args['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
        # args['rpn_bbox_pred_weight'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_weight'])
        args['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        args['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        args['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['bbox_pred_weight'])
        args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # prepare training
    if config.TRAIN.FINETUNE:
        fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    else:
        fixed_param_prefix = ['conv1', 'conv2']
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # only consider rcnn loss
    eval_metric = AccuracyMetric()
    cls_metric = LogLossMetric()
    bbox_metric = SmoothL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005, ##  TODO (use proper wd)
                        'learning_rate': 0.001,   # TODO(use proper lr)
                        'lr_scheduler': mx.lr_scheduler.FactorScheduler(50000, 0.1),
                        'rescale_grad': (1.0 / config.TRAIN.IMS_PER_BATCH)}

    # train
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)
    # import pdb; pdb.set_trace()
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kv_store,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=num_epoch)
    import pdb; pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--test_image_set', dest='test_image_set', help='can be test or val',
                        default='test', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'faster-rcnn'), type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='end epoch of faster rcnn end2end training',
                        default=7, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='local', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    # ctx = [mx.cpu()]
    end2end_train(args.image_set, args.test_image_set, args.year, args.root_path, args.devkit_path,
                  args.pretrained, args.load_epoch, args.prefix, ctx, args.begin_epoch, args.num_epoch,
                  args.frequent, args.kv_store, args.work_load_list)
