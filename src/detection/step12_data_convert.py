#!/usr/bin/env python3
# -*- coding: utf8 -*-

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import os

from util_loaddata import load_folder_images, load_image_file
from util_cache_func import cache_it

# convert data 只是一个函数即可
'''
转换小票图片及label，到文件定位后的图片
'''

def convert_to_box_label_images(src_folder, target_folder):

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for key, image, boxes in load_folder_images(src_folder):

        for idx, (box, label) in enumerate(boxes):

            crop_img = image[box[0][1]:box[2][1], box[0][0]:box[2][0]]

            file_img_name = os.sep.join(
                [target_folder, "{}-{}".format(key, idx)])
            cv2.imwrite(file_img_name + ".jpg", crop_img)

            with open(file_img_name + ".txt", "w+") as fh:
                fh.write(label)

    return

@cache_it()
def load_label_images(src_folder):
    '''
    return: list of,
        image, label text
    '''
    data_file = {}
    label_file = {}

    for root, _, files in os.walk(src_folder, topdown=False):

        for name in files:
            if name.endswith(".txt"):
                label_file[name[:-4]] = root
            elif name.endswith(".jpg"):
                data_file[name[:-4]] = root

    results = []
    for ele in data_file:
        if ele not in label_file:
            print("WARNING: {} is not found with a label.".format(ele), file=sys.stderr)
            continue

        _, image = load_image_file( os.sep.join([data_file[ele], ele+".jpg"]) )

        with open(os.sep.join([label_file[ele], ele+".txt"]), "rt") as fh:
            label = fh.read()

        results.append([image, label])

    return results




# # target folder
if __name__ == "__main__":

    for flder in [
                  "0325-task1-sub000",
                #   "0325-task1-subset1",
                #   "0325-task1-subset2",
                #   "0325-task1-subset3",
                  ]:

        convert_to_box_label_images( os.sep.join(["/data/9.work/ctpn-data/ICDAR2019", flder]),
                                     os.sep.join(["data", flder]) )


    res = load_label_images("data/0325-task1-sub000")
    print( len(res) )
