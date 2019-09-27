# -*- coding: utf-8 -*-
"""
   File Name：     config
   Description :  配置类
   Author :       mick.yi
   date：          2019/3/14
"""

import re
import os

class Config(object):

    IMAGES_PER_GPU = 4

    IMAGE_SHAPE = (720, 720, 3)
    MAX_GT_INSTANCES = 700

    NUM_CLASSES = 1 + 1  #
    CLASS_MAPPING = {'bg': 0,
                     'text': 1}
    # 训练样本
    ANCHORS_HEIGHT = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    ANCHORS_WIDTH = 16
    TRAIN_ANCHORS_PER_IMAGE = 128
    ANCHOR_POSITIVE_RATIO = 0.5
    # 步长
    NET_STRIDE = 16
    # text proposal输出
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    TEXT_PROPOSALS_MAX_NUM = 500
    TEXT_PROPOSALS_WIDTH = 16
    # text line boxes超参数
    LINE_MIN_SCORE = 0.7
    MAX_HORIZONTAL_GAP = 50
    TEXT_LINE_NMS_THRESH = 0.3
    MIN_NUM_PROPOSALS = 1
    MIN_RATIO = 1.2
    MIN_V_OVERLAPS = 0.7
    MIN_SIZE_SIM = 0.7

    # 训练超参数
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9
    # 权重衰减
    WEIGHT_DECAY = 0.0005,
    GRADIENT_CLIP_NORM = 5.0

    LOSS_WEIGHTS = {
        "ctpn_regress_loss": 1.,
        "ctpn_class_loss": 1,
        "side_regress_loss": 1
    }
    # 是否使用侧边改善
    USE_SIDE_REFINE = True
    # 预训练模型

    # folder structure DATA_ROOT
    '''
    ctpn-data
    ├── ICDAR_2015
    │   ├── test_images
    │   ├── train_gt
    │   └── train_images
    ├── ICDAR2019
    │   ├── 0325updated.task1train(626p)
    │   ├── 0325updated.task2train(626p)
    │   ├── fulltext_test(361p)
    │   └── task3-test£¨347p)
    ├── ctpn.100.h5
    └── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    '''
    DATA_ROOT = ["ctpn-data/","/data/9.work/ctpn-data"]

    # in.
    # PRE_TRAINED_WEIGHT = '{}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(DATA_ROOT)

    # # 数据集路径, in
    # IMAGE_DIR = {2015: '{}/ICDAR_2015/train_images'.format(DATA_ROOT),
    #                 2019: '{}/ICDAR2019/0325updated.task1train(626p)/train_images'.format(DATA_ROOT) }

    # IMAGE_GT_DIR = {2015 : '{}/ICDAR_2015/train_gt'.format(DATA_ROOT),
    #                 2019:'{}/ICDAR2019/0325updated.task1train(626p)/train_gt'.format(DATA_ROOT) }

    # # in, out
    # WEIGHT_PATH = '{}/ctpn.100.h5'.format(DATA_ROOT)

    def set_root(self, root):
        if root not in self.DATA_ROOT:
            self.DATA_ROOT.insert(0, root)

    root_dir = None
    def get_root(self):
        if self.root_dir:
            return self.root_dir

        if not isinstance( self.DATA_ROOT, list ):
            self.root_dir = self.DATA_ROOT
            return self.root_dir

        for ele in self.DATA_ROOT:
            if os.path.isdir(ele):
                self.root_dir = ele
                break

        return self.root_dir


    def get_weight_file(self, parent_limit=3):
        '''
        parent_limit: how many levels to go up if not found in current root folder.
            The current folder is not included.
        '''
        re_filter = re.compile(r"ctpn\.(\d+)\.h5")

        res50_exist = None
        cnter = -1
        filename = None

        # abs path is necessary to go up to parent folder.
        cur_folder = os.path.abspath(self.get_root())

        while parent_limit >= 0:
            for file in os.listdir( cur_folder ):
                if not file.endswith(".h5"):
                    continue

                if file.startswith("resnet50"):
                    res50_exist = file
                    continue

                res =  re_filter.match(file)
                if res is None:
                    continue

                if int(res.group(1)) > cnter:
                    cnter = int(res.group(1))

            if cnter != -1:
                filename = "ctpn.{:03d}.h5".format(cnter)
                break

            if res50_exist:
                filename = res50_exist
                break

            cur_folder = os.sep.join( cur_folder.split(os.sep)[:-1] )
            parent_limit -= 1

        if filename:
            return os.sep.join( [cur_folder, filename] )

        return None


    # year_folder = {2015: "ICDAR_2015", 2019: "ICDAR2019"}
    def __getattribute__(self, item):
        '''
        # 不再需要了，只是参考
        if item == "IMAGE_DIR":
            if self.year == 2019:
                return os.sep.join([self.get_root(), self.year_folder[self.year], "0325updated.task1train(626p)", "train_images"])
            else:
                return os.sep.join([self.get_root(), self.year_folder[self.year], "train_images"])
        elif item == "IMAGE_GT_DIR":
            if self.year == 2019:
                return os.sep.join([self.get_root(), self.year_folder[self.year], "0325updated.task1train(626p)", "train_gt"])
            else:
                return os.sep.join([self.get_root(), self.year_folder[self.year], "train_gt"])
        '''
        if item == "WEIGHT_PATH":
            return self.get_weight_file()

        elif item == "PRE_TRAINED_WEIGHT":
            return os.sep.join([self.get_root(), "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"])

        return super(Config, self).__getattribute__(item)

cur_config = Config()
