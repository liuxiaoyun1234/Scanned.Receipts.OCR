
from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import sys
import os

from util_cache_func import cache_it

'''
read image from array.

#read the data from the file
with open(somefile, 'rb') as infile:
     buf = infile.read()

#use numpy to construct an array from the bytes
x = np.fromstring(buf, dtype='uint8')

#decode the array into an image
img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

#show it
cv2.imshow("some window", img)
cv2.waitKey(0)
'''


@cache_it()
def load_image_file(file, max_h=None, max_w=None):

    image = cv2.imread( file )

    h = image.shape[0]
    w = image.shape[1]

    ratio = 1.0
    if max_h and max_w and h > max_h:
        ratio = max_h/h

        w = int(w*ratio)
        h = int(h*ratio)

        image = cv2.resize(image, (w, h))

    return ratio, image

# 小文件，压缩的意义不大
@cache_it(compress=False)
def _load_label_file(file, ratio = 1.0):

    boxes = []

    with open( file ) as fh:
        data_lines = fh.readlines()

        for line in data_lines:
            line_data = line.strip().split(",")
            label = line_data[-1]
            line_data = line_data[:8]
            if len(line_data) != 8:
                print("unknown format", file, line)
                continue

            box = []
            for off in range(0,8,2):
                x, y = line_data[off:off+2]
                x = int(x) * ratio
                y = int(y) * ratio
                box.append( (int(x), int(y)) )

            boxes.append([box, label])

    return boxes


def _load_annotation_single(annotation_path, image_file):
    """
    加载标注信息
    :param annotation_path:
    :param image_file:
    :return:
    """
    image_annotation = {}

    # image_name = base_name  # 通配符 gt_img_3.txt,img_3.jpg or png
    image_annotation["annotation_path"] = annotation_path
    image_annotation["image_path"] = image_file
    image_annotation["file_name"] = os.path.basename(image_annotation["image_path"])  # 图像文件名

    # 读取边框标注
    bbox = []
    quadrilateral = []  # 四边形

    with open(annotation_path, "r", encoding='utf-8') as f:
        lines = f.read().encode('utf-8').decode('utf-8-sig').splitlines()
        # lines = f.readlines()
        # print(lines)
    for line in lines:
        line = line.strip().split(",")
        # 左上、右上、右下、左下 四个坐标 如：377,117,463,117,465,130,378,130
        lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = map(float, line[:8])
        x_min, y_min, x_max, y_max = min(lt_x, lb_x), min(lt_y, rt_y), max(rt_x, rb_x), max(lb_y, rb_y)
        bbox.append([y_min, x_min, y_max, x_max])
        quadrilateral.append([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y])

    image_annotation["boxes"] = np.asarray(bbox, np.float32).reshape((-1, 4))
    image_annotation["quadrilaterals"] = np.asarray(quadrilateral, np.float32).reshape((-1, 8))
    image_annotation["labels"] = np.ones(shape=(len(bbox)), dtype=np.uint8)
    return image_annotation


@cache_it()
def load_folder_annotation(folder):
    '''
    this is from reader.py and udpated.
    '''
    data_file = {}
    label_file = {}

    for root, _, files in os.walk(folder, topdown=False):
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

        results.append( _load_annotation_single(os.sep.join([label_file[ele], ele+".txt"]),
                        os.sep.join([data_file[ele], ele+".jpg"])) )

    return results


@cache_it()
def load_folder_images(folder):
    '''
    input:
        "folder" which contains
            - .txt for box&label
            - .jpg for images

    return: list of,

        key (filename without path or extension), data_image, list of box&label
    '''
    data_file = {}
    label_file = {}

    for root, _, files in os.walk(folder, topdown=False):

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

        _, data = load_image_file( os.sep.join([data_file[ele], ele+".jpg"]) )

        boxes = _load_label_file( os.sep.join([label_file[ele], ele+".txt"]) )

        results.append([ele, data, boxes])

    return results


if __name__ == "__main__":
    # res = load_folder_images("/data/9.work/ctpn-data/ICDAR2019/0325-task1-subset1")
    # print( len(res) )
    res = load_folder_images("/data/9.work/ctpn-data/ICDAR2019/0325-task1-sub000")
    print( len(res) )

