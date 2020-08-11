# coding=UTF-8
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import math
import cv2
import time
import copy
import imageio
import numpy as np
from glob import glob
import skimage
from skimage import io,data,color,morphology,feature,measure

class HpInference:
    def __init__(self):
        self.ori_image = None
        self.image_shape = None

    def gt_erode_img(self, label_path):
        label_name = os.path.basename(label_path)
        groundtruth_name = label_name
        groundtruth_path = label_path.split(labelpath)[0] + groundtruthpath + '/' + groundtruth_name

        label_img = imageio.imread(label_path)
        groundtruth_img = imageio.imread(groundtruth_path)
        gt_erode_img = label_img//255*groundtruth_img
        gt_erode_img = np.array(gt_erode_img,np.uint8)

        if not os.path.exists(bulidpath):
            os.makedirs(bulidpath)
        imageio.imwrite(os.path.join(bulidpath, label_name), gt_erode_img)


def inference(inference_path):
    pngfile = glob(os.path.join(inference_path, '*.png'))
    hp_inference = HpInference()
    for i in range(len(pngfile)):
        label_path=pngfile[i]
        print("pngfile[{}]".format(i),pngfile[i])
        hp_inference.gt_erode_img(label_path=label_path)


if __name__ == '__main__':
    labelpath = 'label_1024_erode'
    groundtruthpath = 'groundtruth_1024'
    bulidgtpath = 'groundtruth_1024_erode'

    for i in range(1,70):
        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_' + str(i)
        if os.path.exists(fatherpath):
            basepath = fatherpath+'/'+labelpath
            bulidpath = fatherpath+'/'+bulidgtpath
            inference(basepath)
        else:
            pass