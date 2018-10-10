# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import darknet_tools
import json
import os
import settings
import glob

from jinja2 import Template
from pythonapi import common_tools
from six.moves import queue


def write_darknet_test_cfg():
    with open('yolo-chinese.template.cfg') as f:
        template = Template(f.read())
    with open(settings.DARKNET_TEST_CFG, 'w') as f:
        f.write(template.render({
            'testing': True,
            'image_size': settings.TEST_IMAGE_SIZE,
            'classes': settings.NUM_CHAR_CATES + 1,
            'filters': 25 + 5 * (settings.NUM_CHAR_CATES + 1),
        }))
        f.write('\n')


def crop_test_images(list_file_name):

    with open(settings.CATES) as f:
        cates = json.load(f)
    text2cate = {c['text']: c['cate_id'] for c in cates}

    if not os.path.isdir(settings.TEST_CROPPED_DIR):
        os.makedirs(settings.TEST_CROPPED_DIR)

    imgPath_list = glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.jpg'))
    imgPath_list += glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.png'))
    testset_size = len(imgPath_list)
    def crop_once(imgPath, testName_list):
        image = cv2.imread(imgPath)
        imshape = image.shape
        fileName = os.path.basename(imgPath)
        image_id = os.path.splitext(fileName)[0]
        image_type = os.path.splitext(fileName)[1]
        cropped_list = []
        for level_id, (cropratio, cropoverlap) in enumerate(settings.TEST_CROP_LEVELS):
            cropshape = (int(round(settings.TEST_IMAGE_SIZE // cropratio)), int(round(settings.TEST_IMAGE_SIZE // cropratio)))
            for o in darknet_tools.get_crop_bboxes(imshape, cropshape, (cropoverlap, cropoverlap)):
                xlo = o['xlo']
                xhi = xlo + cropshape[1]
                ylo = o['ylo']
                yhi = ylo + cropshape[0]
                basename = '{}_{}_{}'.format(image_id, level_id, o['name'])
                cropped_file_name = os.path.join(settings.TEST_CROPPED_DIR, basename + image_type)
                cropped_list.append(cropped_file_name)
                cropped = image[ylo:yhi, xlo:xhi]
                cv2.imwrite(cropped_file_name, cropped)
        testName_list += cropped_list

    q_i = queue.Queue()
    q_i.put(0)

    def foo(*args):
        i = q_i.get()
        if i % 100 == 0:
            print('crop test', i, '/', testset_size)
        q_i.put(i+1)
        crop_once(*args)
    testName_list = []
    common_tools.multithreaded(foo, [(imgPath, testName_list) for imgPath in imgPath_list], num_thread = 4)
    with open(list_file_name, 'w') as f:
        for file_name in testName_list:
            f.write(file_name)
            f.write('\n')


def main():
    write_darknet_test_cfg()
    crop_test_images(settings.DARKNET_VALID_LIST)


if __name__ == '__main__':
    main()
