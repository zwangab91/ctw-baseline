# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import cv2
from PIL import ImageFont, ImageDraw, Image
import settings
import math
import sys
import collections


def bbox2poly(bbox):
    '''
    convert unadjusted bbox to polygon
    bbox: [xmin, ymin, w, h]
    polygon: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] is the original bbox
    '''	
    xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xmax, ymax = xmin + w/1.0, ymin + h/1.0
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
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	yoloPred_path = os.path.join(curr_dir, 'products', 'detections.jsonl')
	par_dir = os.path.dirname(curr_dir)
	labeled_save_dir = os.path.join(par_dir, 'data', 'images', 'test', 'yolo_v2') 

	with open(yoloPred_path, 'r') as f:
		for line in f:
			data = json.loads(line)
			det = data['detections']
			above_thres_det = list(filter(lambda x: x['score'] >= settings.DET_THRES, det))
			file_name = data['file_name']
			file_path = os.path.join(settings.TEST_IMAGE_DIR, file_name)
			imwrite_path = os.path.join(labeled_save_dir, file_name)
			image = cv2.imread(file_path)
			font_path = '/System/Library/Fonts//PingFang.ttc'
			font = ImageFont.truetype(font_path, 15)

			for char in above_thres_det:
				bbox = char['bbox']
				text = char['text']
				#prob = char['score']

				polygon = bbox2poly(bbox)
				adjusted_bbox = poly2bbox(polygon)[1]
				xmin, ymin, w, h = adjusted_bbox
				xmax, ymax = xmin + w, ymin + h
				cv2.rectangle(image, (int(math.floor(xmin)),int(math.floor(ymin))), 
					(int(math.ceil(xmax)),int(math.ceil(ymax))), 2)
				image_pil = Image.fromarray(image)
				draw = ImageDraw.Draw(image_pil)
				draw.text((int(math.floor(xmin)),int(math.ceil(ymax))), text, 
					font = font, fill = (0, 0, 255))
				image = np.array(image_pil)
			cv2.imwrite(imwrite_path, image)


