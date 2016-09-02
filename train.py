import argparse
import logging
import os

import sys
sys.path.insert(0, '/home/work/wuwei/project/dmlc/mxnet/python')
import mxnet as mx

from rcnn.callback import Speedometer
from rcnn.config import config
from rcnn.loader import AnchorLoader
from rcnn.metric import AccuracyMetric, LogLossMetric, SmoothL1LossMetric
from rcnn.module import MutableModule
from rcnn.symbol import get_faster_rcnn
from utils.load_data import load_gt_roidb_from_list
from utils.load_model import do_checkpoint, load_param
from rcnn.warmup import WarmupScheduler

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config.TRAIN.BG_THRESH_HI = 0.5  # TODO(verify)
    config.TRAIN.BG_THRESH_LO = 0.0  # TODO(verify)
    config.SCALES = (600, )
    config.MAX_SIZE = 1000
    config.TRAIN.RPN_MIN_SIZE = 10

    logging.info('########## TRAIN FASTER-RCNN WITH APPROXIMATE JOINT END2END #############')
    config.TRAIN.HAS_RPN = True
    config.END2END = 1
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    sym = get_faster_rcnn(num_classes=args.num_classes)  # consider background
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    config.TRAIN.IMS_PER_BATCH *= len(ctx)
    config.TRAIN.RPN_BATCH_SIZE *= len(ctx)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.IMS_PER_BATCH, 3, 1000, 1000))]
    max_data_shape_dict = {k: v for k, v in max_data_shape}
    _, feat_shape, _ = feat_sym.infer_shape(**max_data_shape_dict)
    from rcnn.minibatch import assign_anchor
    import numpy as np
    label = assign_anchor(feat_shape[0], np.zeros((0, 5)), [[1000, 1000, 1.0]])
    max_label_shape = [('label', label['label'].shape),
                       ('bbox_target', label['bbox_target'].shape),
                       ('bbox_inside_weight', label['bbox_inside_weight'].shape),
                       ('bbox_outside_weight', label['bbox_outside_weight'].shape),
                       ('gt_boxes', (config.TRAIN.RPN_BATCH_SIZE, 5*200))]  # assume at most 200 faces in image
    # print 'providing maximum shape', max_data_shape, max_label_shape

    voc, roidb = load_gt_roidb_from_list(args.dataset_name, args.lst, args.dataset_root,
                                         args.outdata_path, flip=not args.no_flip)
    train_data = AnchorLoader(feat_sym, roidb, batch_size=config.TRAIN.IMS_PER_BATCH, shuffle=not args.no_shuffle,
                              mode='train', ctx=ctx)
    args_params, auxs_params, _ = load_param(args.pretrained, args.load_epoch, convert=True)

    if not args.resume:
        del args_params['fc8_weight']
        del args_params['fc8_bias']
        input_shapes = {k: (1,)+ v[1::] for k, v in train_data.provide_data + train_data.provide_label}
        arg_shape, _, _ = sym.infer_shape(**input_shapes)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))

        args_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        args_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        args_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        args_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        args_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['rpn_bbox_pred_weight'])  # guarantee not likely explode with bbox_delta
        args_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        args_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        args_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        args_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['bbox_pred_weight'])
        args_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # prepare training
    if config.TRAIN.FINETUNE:
        fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    else:
        fixed_param_prefix = ['conv1', 'conv2']
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    batch_end_callback = Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = do_checkpoint(args.prefix)
    eval_metric = AccuracyMetric()
    cls_metric = LogLossMetric()
    bbox_metric = SmoothL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    optimizer_params = {'momentum': args.mom,
                        'wd': args.wd,
                        'learning_rate': args.lr,
                        'lr_scheduler': WarmupScheduler(args.factor_step, 0.1, warmup_lr=0.1*args.lr, warmup_step=200) \
                                        if not args.resume else mx.lr_scheduler.FactorScheduler(args.factor_step, 0.1),
                        'clip_gradient': 1.0,
                        'rescale_grad': (1.0 / config.TRAIN.RPN_BATCH_SIZE)}
    # train
    mod = MutableModule(sym, data_names=data_names, label_names=label_names, logger=logger, context=ctx,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kv_store,
            optimizer='sgd', optimizer_params=optimizer_params, arg_params=args_params, aux_params=auxs_params,
            begin_epoch=args.load_epoch, num_epoch=args.num_epoch)

if __name__ == '__main__':
    logging.info('############### TRAIN FASTER-RCNN WITH APPROXIMATE JOINT END2END ##################\n'
                 '          -----------------------------------------------------------------------------------')
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network using list file of annotation')
    parser.add_argument('--dataset-name', type=str, default='wider_face', help='give a name of your dataset')
    parser.add_argument('--image_set', type=str, default='trainval', help='can be trainval or train')
    parser.add_argument('--test_set', type=str, default='test', help='can be test or val')
    parser.add_argument('--lst', dest='lst', type=str, default='data/trainval.lst', help='the list file of annotation')
    parser.add_argument('--num-classes', dest='num_classes', help='the class number of detection, include backgound',
                        default=2, type=int)
    parser.add_argument('--outdata-path', type=str, default=os.path.join(os.getcwd(), 'data'),
                        help='output data folder')
    parser.add_argument('--dataset-root', type=str, default=os.path.join(os.getcwd(), 'data'),
                        help='the root path of your dataset')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'faster-rcnn'), type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--num_epoch', dest='num_epoch', help='end epoch of faster rcnn end2end training',
                        default=7, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv-store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--no-flip', action='store_true', default=False,
                        help='if true, then will flip the dataset')
    parser.add_argument('--no-shuffle', action='store_true', default=False,
                        help='if true, then will shuffle the dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay for sgd')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='if true, then will retrain the model from rcnn')
    parser.add_argument('--factor-step',type=int, default=50000, help='the step used for lr factor')
    args = parser.parse_args()
    logging.info(args)
    print "\n          -----------------------------------------------------------------------------------"
    main()
