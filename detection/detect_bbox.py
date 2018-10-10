# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ctypes import *
import os
import random
import cv2
import settings
import darknet_tools
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import copy
import json
import collections
import glob

def c_array(ctype, values):
	return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


curr_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(curr_dir)
lib_path = os.path.join(curr_dir, 'darknet', 'libdarknet.so')
lib = CDLL(lib_path, RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def yolo2poly(yolo):
    '''
    convert yolo bbox to polygon
    yolo: [xmid, ymid, w, h]
    polygon: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] is the original bbox
    '''
    xmid, ymid, w, h = yolo[0], yolo[1], yolo[2], yolo[3]
    xmin, xmax = xmid - w/2.0, xmid + w/2.0
    ymin, ymax = ymid - h/2.0, ymid + h/2.0
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def poly2bbox(poly):
    '''
    convert polygon to bbox
    polygon: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] is the original bbox
    bbox: [xmin, ymin, w, h] is the adjusted_bbox
    '''
    key_points = list()
    rotated = collections.deque(poly)
    rotated.rotate(1)
    for (x0, y0), (x1, y1) in zip(poly, rotated):
        for ratio in (1/3, 2/3):
            key_points.append([x0 * ratio + x1 * (1 - ratio), y0 * ratio + y1 * (1 - ratio)])
    x, y = zip(*key_points)
    adjusted_bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]
    return key_points, adjusted_bbox


if __name__ == '__main__':

    cfg_path = os.path.join(curr_dir, 'darknet', 'cfg', 'yolo-chinese.cfg')
    weight_path = os.path.join(curr_dir, 'darknet', 'weights', 'yolo-chinese_final.weights')
    dataFile_path = os.path.join(curr_dir, 'darknet', 'cfg', 'chinese.data')
    net = load_net(cfg_path, weight_path, 0)
    meta = load_meta(dataFile_path)

    save_path = os.path.join(par_dir, 'data', 'annotations', 'test_cls.jsonl')
    data_dir = os.path.join(par_dir, 'data', 'images', 'test')
    dataPath_list = glob.glob(os.path.join(data_dir, '*.jpg'))
    dataPath_list += glob.glob(os.path.join(data_dir, '*.png'))
    with open(save_path, 'w') as f:
        percent = 0.0
        for dataPath in dataPath_list:
            filename = os.path.basename(dataPath)
            # darknet
            #r = detect(net, meta, data_path)
            # opencv
            arr = cv2.imread(dataPath)
            height, width, channels = arr.shape
            im = array_to_image(arr)
            rgbgr_image(im)
            r = detect2(net, meta, im)

            dic = {'file_name': filename, 
            'height': height,
            'image_id': os.path.splitext(filename)[0]}
            proposals = []
            for anno in r:
                yolo = anno[2]
                polygon = yolo2poly(yolo)
                bbox = poly2bbox(polygon)[1]
                proposals.append({'adjusted_bbox': bbox, 
                    'polygon': polygon})
            dic['proposals'] = proposals
            dic['width'] = width
            f.write(json.dumps(dic))
            f.write('\n')
            dic = None

            percent += 100 / len(dataPath_list)
            print('finished %f percent'%percent)
