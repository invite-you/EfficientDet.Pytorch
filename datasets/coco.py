from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image

import pickle
import math 
import pandas as pd
import os

class CocoDataset(Dataset):
    """Coco dataset."""
    def __init__(self, pickle_file, transform=None):

        self.root_dir = os.path.dirname(pickle_file)
        self.transform = transform
        
        self.rbboxs = []
        self.image_container = []

        self.parse_pickle(pickle_file)
        print("## all dataset: ", len(self.rbboxs))

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = [{'id': 1, 'name': 'container'}, {'id': 2, 'name': 'oil tanker'}, {'id': 3, 'name': 'aircraft carrier'}, {'id': 4, 'name': 'maritime vessels'}]
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def parse_pickle(self, pickle_file):
      
        rbboxs = []
        image_container = []
        idx = -1

        with open(pickle_file, 'rb') as f:
            pickle_info = pickle.load(f)

        for tidx in range(len(pickle_info)):

            image_id = pickle_info[tidx]['image_filename']
            row = pickle_info[tidx]['row']
            col = pickle_info[tidx]['col']
            patch_size = pickle_info[tidx]['patch_size']

            filename = "_".join([image_id.replace(".png", ''), str(row), str(col)]) + '.png'
            filepath = os.path.join(self.root_dir, 'patch_images/')
            
            if not os.path.isfile(os.path.join(filepath, filename)):
                print("ERROR havn't file :", os.path.join(filepath, filename))
                continue
                
            idx += 1
            try:
                self.image_container[idx]
            except IndexError:
                self.image_container.append(filename)

            cxs, cys = pickle_info[idx]['center_xs'], pickle_info[idx]['center_ys']
            hs, ws = pickle_info[idx]['heights'], pickle_info[idx]['widths']
            thetas, labels = pickle_info[idx]['thetas'], pickle_info[idx]['class_indices']

            for cx, cy, h, w, theta, label in zip(cxs, cys, hs, ws, thetas, labels):
                heigth = pickle_info[idx]['patch_height']
                width = pickle_info[idx]['patch_width']
                (imgcX, imgcY) = (width // 2, heigth // 2)
                theta= math.degrees(theta)

                cy = cy * heigth
                cx = cx * width
                h = h *heigth
                w = w * width

                rbox = self.convert_bbox_to_rbox(cx, cy, h, w, theta)

                rbbox = self.rotate_box(rbox, imgcX, imgcY, heigth, width, theta)
                
                # 이미지 회전 후 bbox가 이미지 사이즈보다 더 커지는 경우 방지
                # bbox가 이미지 사이즈 보다 크거나 작을 경우 사이즈를 옮긴다
                xs = [box[0] for box in rbbox]
                ys = [box[1] for box in rbbox]

                x_diff = self.diff_baseline_image(xs, width)
                y_diff = self.diff_baseline_image(ys, heigth)

                rbbox = [[x+x_diff, y+y_diff] for x, y in zip(xs, ys)]

                xy = pd.DataFrame(data=rbbox, columns=['x','y'])
                xy1 = xy.nsmallest(2, 'x').nsmallest(1, 'y').values[0].tolist()    
                xy2 = xy.nlargest(2, 'y').nlargest(1, 'x').values[0].tolist()  

                """
                if xy1[0] <= 0:
                    xy1[0] = 1
                if width <= xy2[0]:
                    xy2[0] = width - 1

                if xy1[1] <= 0:
                    xy1[1] = 1
                if heigth <= xy2[1]:
                    xy2[1] = heigth-1
                """

                self.rbboxs.append(
                    {"image_param": {'x_diff': x_diff, 'y_diff': y_diff,
                                    'theta': theta, 'fileidx': idx},
                    "rbbox": {'minX': xy1[0], 'minY': xy1[1],
                                'maxX': xy2[0], 'maxY': xy2[1],
                                'label': label}
                    })
        

    def convert_bbox_to_rbox(self, cx, cy, h, w, theta):
        def rotate_init_box(bb, cx, cy, theta):
            new_bb = list(bb)
            for i,coord in enumerate(bb):
                # opencv calculates standard transformation matrix
                M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
                # Grab  the rotation components of the matrix)
                # Prepare the vector to be transformed
                v = [coord[0],coord[1],1]
                # Perform the actual rotation and return the image
                calculated = np.dot(M,v)
                new_bb[i] = (calculated[0],calculated[1])
            return new_bb
        x1, y1, x3, y3 = cx-(w/2), cy-(h/2), cx+(w/2), cy+(h/2)
        x4, y4, x2, y2 = cx-(w/2), cy+(h/2), cx+(w/2), cy-(h/2)
        bbox = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
        rbox = rotate_init_box(bbox, cx, cy, -theta)
        return rbox

    def rotate_box(self, bb, cx, cy, h, w, theta):
        new_bb = list(bb)
        for i,coord in enumerate(bb):
            # opencv calculates standard transformation matrix
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
            # Prepare the vector to be transformed
            v = [coord[0],coord[1],1]
            # Perform the actual rotation and return the image
            calculated = np.dot(M,v)
            new_bb[i] = (calculated[0],calculated[1])
        return new_bb

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        #M[0, 2] += (nW / 2) - cX
        #M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def diff_baseline_image(self, points, base):
        points = np.array(points)
        padding = 0
        #print("smallest: ", points[points < 0])
        #print("biggest: ", points[base < points])

        if (not len(points[base < points]) == 0) and (not len(points[points < 0]) == 0):
            print("What??? small image")
            return 0
        diff=0
        over = points[base < points]
        if not 0 == len(over):
            diff = max(over) - base + padding

        over = points[points < 0]
        if not 0 == len(over):
            diff = min(over) - padding
        return -diff      

    def __len__(self):
        return len(self.rbboxs)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        
        x_diff, y_diff = self.rbboxs[image_index]['image_param']['x_diff'], self.rbboxs[image_index]['image_param']['y_diff']
        theta, fileidx = self.rbboxs[image_index]['image_param']['theta'], self.rbboxs[image_index]['image_param']['fileidx']
        image_index = self.rbboxs[image_index]['image_param']['fileidx']

        image_path = os.path.join(self.root_dir, 'patch_images/', self.image_container[image_index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 돌리기
        rimage = self.rotate_bound(image, theta)
        # 이미지 이동
        (heigth, width) = image.shape[:2]
        #print(heigth, width)
        M = np.float32([[1,0,x_diff],[0,1,y_diff]])
        rimage = cv2.warpAffine(rimage, M, (width, heigth))
        return rimage


    def load_annotations(self, image_index):
        # get ground truth annotations
        coco_annotations = [self.rbboxs[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(coco_annotations) == 0:
            return annotations

        # parse annotations
        for idx, annot in enumerate(coco_annotations):

            minX = float(annot['rbbox']['minX'])
            minY = float(annot['rbbox']['minY'])
            maxX = float(annot['rbbox']['maxX'])
            maxY = float(annot['rbbox']['maxY'])

            # some annotations have basically no width / height, skip them
            if maxX < 1 or maxY < 1:
                continue
            
            if minX < 0 : minX = 1
            if minY < 0 : minY = 1
            if maxX > 1024 : minX = 1023
            if maxY > 1024 : minY = 1023
            
            annotation = np.zeros((1, 5))
            annotation[0, 0] = minX
            annotation[0, 1] = minY
            annotation[0, 2] = maxX
            annotation[0, 3] = maxY
            annotation[0, 4] = annot['rbbox']['label']
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        #image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return 1024/1024#float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.classes)


if __name__ == '__main__':
    from augmentation import get_augumentation
    dataset = CocoDataset("/content/gdrive/My Drive/findShip/test.pickle",
                          transform=get_augumentation(phase='train', width=1024, height=1024))
    sample = dataset[0]
    print('sample: ', sample)
