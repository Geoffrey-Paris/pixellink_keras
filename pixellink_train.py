#-*- coding:utf-8 -*-
#'''
# Created on 18-10-15
#
# @Author: Greg Gao(laygin)
#'''
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils

from pixellink_model import create_pixellink_model
from pixellink_utils import *

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    NB_IMG = 5
    PATH_TO_DIR = r'./dataset/train/'


    list_dir = glob.glob(PATH_TO_DIR+'*.jpg')
    save_weights = r'./weights/pixellink.h5'
    model = create_pixellink_model(acf='relu')
    model.load_weights(save_weights)

    x = []
    for img_path in list_dir[:NB_IMG]:
        tmp = cv2.imread(img_path)
        image, *r = resize_image(tmp)

        # image = image[...,::-1] - rgb_mean
        image = np.expand_dims(image, axis=0)
        x.append(image.copy())

    pixel_pos_scores_tab, link_pos_scores_tab = model.predict(x)
    for pixel_pos_scores, link_pos_scores, image_c in zip(pixel_pos_scores_tab, link_pos_scores_tab, x):
        pixel_pos_scores = softmax(pixel_pos_scores, axis=-1)
        link_pos_scores_reshaped = link_pos_scores.reshape(link_pos_scores.shape[:-1]+(8, 2))
        link_pos_scores = softmax(link_pos_scores_reshaped, axis=-1)

        masks = decode_batch(pixel_pos_scores, link_pos_scores, pixel_conf_threshold=0.75, link_conf_threshold=0.9)

        bboxes = mask_to_bboxes(masks[0], image_c.shape)


        # image_c = image_ori.copy()
        for box in bboxes:
            points = np.reshape(box, [4, 2])
            cv2.line(image_c,tuple(points[0]),tuple(points[1]),(0,0,255),2)
            cv2.line(image_c,tuple(points[0]),tuple(points[3]),(0,0,255),2)
            cv2.line(image_c,tuple(points[1]),tuple(points[2]),(0,0,255),2)
            cv2.line(image_c,tuple(points[2]),tuple(points[3]),(0,0,255),2)



        cv2.imwrite(PATH_TO_DIR + list_dir, image_c)
