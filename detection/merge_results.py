# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import darknet_tools
import functools
import json
import os
import settings
import subprocess
import sys
import cv2
import glob

from collections import defaultdict
from pythonapi import common_tools, eval_tools
from six.moves import cPickle


def read(file_paths):
    all = defaultdict(list)
    removal = (0., 0., 0.)
    size_ranges = ((float('-inf'), float('inf')), (32., float('inf')), (64., float('inf')))

    # construct imshape dict
    imshape_dic = {}
    testPath_list = glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.jpg'))
    testPath_list += glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.png'))
    for testPath in testPath_list:
        image = cv2.imread(testPath)
        fileName = os.path.basename(testPath)
        imshape = image.shape
        imshape_dic[fileName] = imshape

    def construct_levelmap(imshape):
        levelmap = {}
        for level_id, (cropratio, cropoverlap) in enumerate(settings.TEST_CROP_LEVELS):
            cropshape = (settings.TEST_IMAGE_SIZE // cropratio, settings.TEST_IMAGE_SIZE // cropratio)
            for o in darknet_tools.get_crop_bboxes(imshape, cropshape, (cropoverlap, cropoverlap)):
                levelmap[level_id, o['name']] = (o['xlo'], o['ylo'], cropshape[1], cropshape[0])
        return levelmap

    def bounded_bbox(bbox, imshape):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(imshape[1], x1), min(imshape[0], y1)
        return (x0, y0, x1 - x0, y1 - y0)

    def read_one(result_file_path, levelmap_dic):
        with open(result_file_path) as f:
            lines = f.read().splitlines()
        one = []
        for line in lines:
            file_path, cate_id, x, y, w, h, prob = line.split()
            image_id, level_id, crop_name = os.path.splitext(os.path.basename(file_path))[0].split('_', 2)
            image_type = os.path.splitext(os.path.basename(file_path))[1]
            level_id = int(level_id)
            # construct the levelmap from the uncropped image and get the position and size of the cropped image
            # cx, cy, cw, ch = o['xlo'], o['ylo'], cropshape[1], cropshape[0]
            imshape = imshape_dic[image_id + image_type]
            if imshape in levelmap_dic:
                levelmap = levelmap_dic[imshape]
            else:
                levelmap = construct_levelmap(imshape)
                levelmap_dic[imshape] = levelmap
            cx, cy, cw, ch = levelmap[level_id, crop_name]

            cate_id = settings.NUM_CHAR_CATES if proposal_output else int(cate_id)
            x, y, w, h, prob = float(x), float(y), float(w), float(h), float(prob)
            longsize = max(w, h)
            size_range = size_ranges[level_id]
            if longsize < size_range[0] or size_range[1] <= longsize:
                continue
            rm = removal[level_id]
            if (cx != 0 and x < rm) or (cy != 0 and y < rm) or (cx + cw != imshape[1] and x + w + rm >= cw) or (cy + ch != imshape[0] and y + h + rm >= ch):
                continue
            real_bbox = bounded_bbox((x + cx, y + cy, w, h), imshape)
            if real_bbox[2] > 0 and real_bbox[3] > 0:
                all[image_id].append({'image_id': image_id, 'cate_id': cate_id, 'prob': prob, 'bbox': real_bbox})
    levelmap_dic = {}
    for file_path in file_paths:
        read_one(file_path, levelmap_dic)
    return all


def do_nms_sort(unmerged, nms):
    all = defaultdict(list)
    i_time = 0
    for image_id, proposals in unmerged.items():
        if i_time % 200 == 0:
            print('nms sort', i_time, '/', len(unmerged))
        i_time += 1
        cates = defaultdict(list)
        for proposal in proposals:
            cates[proposal['cate_id']].append(proposal)
        for cate_id, proposal in cates.items():
            a = sorted(proposal, key=lambda o: -o['prob'])
            na = []
            for o in a:
                covered = 0
                for no in na:
                    covered += eval_tools.a_in_b(o['bbox'], no['bbox'])
                    if covered > nms:
                        break
                if covered <= nms:
                    na.append(o)
                    if len(na) >= settings.MAX_DET_PER_IMAGE:
                        break
            cates[cate_id] = na
        all[image_id] = common_tools.reduce_sum(cates.values())
    return all


def write(nms_sorted, file_path):
    with open(settings.CATES) as f:
        cates = json.load(f)

    testPath_list = glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.jpg'))
    testPath_list += glob.glob(os.path.join(settings.TEST_IMAGE_DIR, '*.png'))
    with open(file_path, 'w') as f:
        for testPath in testPath_list:
            fileName = os.path.basename(testPath)
            image_id = os.path.splitext(fileName)[0]
            detections = nms_sorted[image_id]
            detections.sort(key=lambda o: (-o['prob'], o['cate_id'], o['bbox']))
            f.write(common_tools.to_jsonl({
                'file_name': fileName,
                'image_id': image_id,
                'detections': [{
                    'text': cates[dt['cate_id']]['text'],
                    'bbox': dt['bbox'],
                    'score': dt['prob'],
                } for dt in detections if dt['cate_id'] < settings.NUM_CHAR_CATES][:settings.MAX_DET_PER_IMAGE],
                'proposals': [{  # for printtext
                    'text': '',
                    'bbox': dt['bbox'],
                    'score': dt['prob'],
                } for dt in detections if dt['cate_id'] == settings.NUM_CHAR_CATES][:settings.MAX_DET_PER_IMAGE],
            }))
            f.write('\n')


def main():
    file_paths = []
    for split_id in range(settings.TEST_SPLIT_NUM):
        darknet_results_out = darknet_tools.append_before_ext(settings.DARKNET_RESULTS_OUT, '.{}'.format(split_id))
        result_file_path = os.path.join(settings.DARKNET_RESULTS_DIR, '{}.txt'.format(darknet_results_out))
        file_paths.append(result_file_path)

    print('loading darknet outputs')
    unmerged = read(file_paths)

    print('doing nms sort')
    nms_sorted = do_nms_sort(unmerged, .5)

    print('writing results')
    write(nms_sorted, os.path.join(settings.PRODUCTS_ROOT, 'proposals.jsonl' if proposal_output else 'detections.jsonl'))


if __name__ == '__main__':
    proposal_output = 'proposal' in sys.argv[1:]
    main()
