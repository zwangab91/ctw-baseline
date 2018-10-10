#####################################################
# Description: Load the YOLO prediction file and convert 
# it to the input of a classification program.
#####################################################

import os
import json
import collections
import settings

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
	par_dir = os.path.dirname(curr_dir)
	yoloPred_path = os.path.join(curr_dir, 'products', 'detections.jsonl')
	cls_path = os.path.join(par_dir, 'data', 'annotations', 'test_cls.jsonl')
	#cls_path = os.path.join(curr_dir, 'products', 'customized', 'customized_test_cls.jsonl')

	with open(yoloPred_path, 'r') as f, open(cls_path, 'w') as g:
		for line in f:
			data = json.loads(line)
			file_name = data['file_name']
			image_id = data['image_id']
			dic = {'file_name': file_name, 'image_id': image_id}

			det = data['detections']
			# filter out the detections with confidence below the threshold
			above_thres_det = list(filter(lambda x: x['score'] >= settings.DET_THRES, det))
			proposals = []
			for item in above_thres_det:
				bbox = item['bbox'] # unadjusted_bbox
				polygon = bbox2poly(bbox)
				adjusted_bbox = poly2bbox(polygon)[1]
				proposals.append({'adjusted_bbox': adjusted_bbox, 
					'polygon': polygon})
				
			'''
			prop = data['proposals']
			above_thres_prop = list(filter(lambda x: x['score'] >= settings.DET_THRES, prop))
			for item in above_thres_prop:
				bbox = item['bbox']
				polygon = bbox2poly(bbox)
				adjusted_bbox = poly2bbox(polygon)[1]
				proposals.append({'adjusted_bbox': adjusted_bbox, 
					'polygon': polygon})
			'''

			dic['proposals'] = proposals
			g.write(json.dumps(dic))
			g.write('\n')
			dic = None


