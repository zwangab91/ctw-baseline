# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import cv2
from PIL import ImageFont, ImageDraw, Image
import math
import sys

def draw_predict(model_name):

	curr_dir = os.path.dirname(os.path.abspath(__file__))
	par_dir = os.path.dirname(curr_dir)
	predFile_dir = os.path.join(curr_dir, 'products', 'predictions_{}.jsonl'.format(model_name))
	testAnno_dir = os.path.join(par_dir, 'data', 'annotations', 'test_cls.jsonl')
	test_image_dir = os.path.join(par_dir, 'data', 'images', 'test')
	labeled_save_dir = os.path.join(par_dir, 'data', 'images', 'test', model_name) 

	with open(predFile_dir, 'r') as f, open(testAnno_dir, 'r') as g:
		for anno_line, pred_line in zip(g, f):
			anno = json.loads(anno_line)
			pred = json.loads(pred_line)

			proposals = anno['proposals']
			predictions = pred['predictions']
			probabilities = pred['probabilities']
			filename = anno['file_name']
			file_path = os.path.join(test_image_dir, filename)
			imwrite_path = os.path.join(labeled_save_dir, filename)
			image = cv2.imread(file_path)
			fontpath = '/System/Library/Fonts//PingFang.ttc'
			font = ImageFont.truetype(fontpath, 15)

			assert len(proposals) == len(predictions)
			numChar = len(predictions)
			for char_id in range(numChar):
				#top5_char = predictions[char_id]
				#top5_prob = probabilities[char_id] 
				top_char = predictions[char_id][0]
				top_prob = probabilities[char_id][0]
				#assert len(top5_char) == len(top5_prob)
				if top_prob >= 0.5:
					
					bbox = proposals[char_id]
					adjusted_bbox = bbox['adjusted_bbox']
					xmin = adjusted_bbox[0]
					ymin = adjusted_bbox[1]
					w = adjusted_bbox[2]
					h = adjusted_bbox[3]
					xmax = xmin + w
					ymax = ymin + h
					cv2.rectangle(image, (int(math.floor(xmin)),int(math.floor(ymin))), (int(math.ceil(xmax)),int(math.ceil(ymax))), 2)
					image_pil = Image.fromarray(image)
					draw = ImageDraw.Draw(image_pil)
					draw.text((int(math.floor(xmin)),int(math.ceil(ymax))), top_char, font = font, fill = (0, 0, 255))
					#draw.text((int(math.floor(xmin)),int(math.ceil(ymax)+15)), '{0:.2f}'.format(top_prob), font = font, fill = (0, 0, 255))
					#vshift = 0
					#hspace = 15
					#for char, prob in zip(top5_char, top5_prob):
					#	draw.text((int(math.floor(xmin)),int(math.ceil(ymax)+vshift)), char, font = font, fill = (0, 0, 255))
					#	draw.text((int(math.floor(xmin)+hspace),int(math.ceil(ymax)+vshift)), '{0:.2f}'.format(prob), font = font, fill = (0, 0, 255))
					#	vshift = vshift + 15
					image = np.array(image_pil)
			cv2.imwrite(imwrite_path, image)


if __name__ == '__main__':
	draw_predict(sys.argv[1])

