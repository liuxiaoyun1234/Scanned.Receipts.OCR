# -*- coding: utf-8 -*-
"""
   File Name：     train
   Description :   ctpn训练
   Author :       mick.yi
   date：          2019/3/14
"""
import os
import re
import sys
import tensorflow as tf
import keras
import argparse

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from ctpn.layers import models
from ctpn.config import cur_config as config
from ctpn.utils import file_utils
from ctpn.utils.generator import generator

from util_loaddata import load_folder_annotation

def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cfg = tf.ConfigProto()
    # cfg = tf.ConfigProto(log_device_placement = True)
    cfg.gpu_options.allow_growth = True
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
    cfg.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc

    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def get_call_back():
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/ctpn.{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 period=5)

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=1e-4)
    log = TensorBoard(log_dir='log')
    return [lr_reducer, checkpoint, log]


def main(args):

    set_gpu_growth()
    config.set_root(args.root)

    image_annotations = load_folder_annotation(args.root)
    if len(image_annotations) < 5:
        print("Too small dataset...")
        return


    # gen = generator(image_annotations[:-100],
    #                 config.IMAGES_PER_GPU,
    #                 config.IMAGE_SHAPE,
    #                 config.ANCHORS_WIDTH,
    #                 config.MAX_GT_INSTANCES,
    #                 horizontal_flip=False,
    #                 random_crop=False)

    # val_gen = generator(image_annotations[-100:],
    #                     config.IMAGES_PER_GPU,
    #                     config.IMAGE_SHAPE,
    #                     config.ANCHORS_WIDTH,
    #                     config.MAX_GT_INSTANCES)


    # for bat in range(100):
    #     print(bat)

    #     val, _ = next(gen)
    #     for key in val.keys():
    #         print( val[key].shape, end="," )
    #     print()
    #     val, _ = next(val_gen)
    #     for key in val.keys():
    #         print( val[key].shape, end="," )

    #     print()
    #     print()
    # exit(1)





    # 加载模型
    m = models.ctpn_net(config, 'train')
    models.compile(m, config, loss_names=['ctpn_regress_loss', 'ctpn_class_loss', 'side_regress_loss'])
    # 增加度量
    output = models.get_layer(m, 'ctpn_target').output
    models.add_metrics(m, ['gt_num', 'pos_num', 'neg_num', 'gt_min_iou', 'gt_avg_iou'], output[-5:])

    # 从0开始的话，用resnet50
    if args.weight_path is None:
        args.weight_path = config.WEIGHT_PATH

    # get current epoch from file name.
    if args.init_epochs == 0:
        res = re.match(r".*ctpn\.(\d+)\.h5", args.weight_path)
        if res is not None:
            args.init_epochs = int(res.group(1))

    m.load_weights( args.weight_path, by_name=True)

    m.summary()

    # print( len( image_annotations[:-100]), len(image_annotations[-100:]) )
    # 生成器
    # 前面100条作为训练集，后面100条做成测试集。

    gen = generator(image_annotations[:-10],
                    config.IMAGES_PER_GPU,
                    config.IMAGE_SHAPE,
                    config.ANCHORS_WIDTH,
                    config.MAX_GT_INSTANCES,
                    horizontal_flip=False,
                    random_crop=False)

    val_gen = generator(image_annotations[-10:],
                        config.IMAGES_PER_GPU,
                        config.IMAGE_SHAPE,
                        config.ANCHORS_WIDTH,
                        config.MAX_GT_INSTANCES)

    # 训练
    m.fit_generator(gen,
                    steps_per_epoch=len(image_annotations) // config.IMAGES_PER_GPU * 2,
                    epochs=args.epochs,
                    initial_epoch=args.init_epochs,
                    validation_data=val_gen,
                    validation_steps=100 // config.IMAGES_PER_GPU,
                    verbose=True,
                    callbacks=get_call_back(),
                    workers=args.jobs,
                    use_multiprocessing=True)

    #

    # # 保存模型
    path = os.path.split(config.WEIGHT_PATH)
    m.save( os.sep.join([path[0], "ctpn.%03d.h5"%(args.epochs)] ) )


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--root", required=True, help="data root folder")
    parse.add_argument("--epochs", type=int, default=100, help="epochs, original defalt 100")
    parse.add_argument("--init_epochs", type=int, default=0, help="epochs")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--jobs", type=int, default=1, help="concurrent jobs")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)
