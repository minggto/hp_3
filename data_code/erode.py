# coding=UTF-8
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
import skimage
from skimage import io,data,color,feature,measure
import cv2
import time
import copy
import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from skimage.measure import regionprops
from skimage.measure import label
from PIL import Image, ImageDraw, ImageFont
#from skimage import io,data,color,morphology,feature,measure

class HpInference:
    def __init__(self):
        
        self.ori_image = None
        self.image_shape = None
        

    def predict(self, image_path):
        groudtruth = imageio.imread(image_path)        
        self._visualization_statis(groudtruth, image_path, visual_path=visualpath_statis)


    def _visualization_statis(self, groudtruth, image_path, visual_path):
        image_name = os.path.basename(image_path)
        
        if not os.path.exists(visual_path+'erode'):
            os.makedirs(visual_path+'erode')
        kernel = np.ones((20,20),np.uint8)
        dilation = cv2.erode(groudtruth,kernel,iterations = 1) #膨胀
        dilat_groudtruth = dilation
        imageio.imwrite(os.path.join(visual_path+'erode', image_name.split('.png')[0] + '.png'), dilat_groudtruth)



def inference(inference_path):
    inference_files = glob(os.path.join(inference_path, '*.png'))
    jpgfile=inference_files
    print("inference_files",jpgfile[0])
    hp_inference = HpInference()
    for i in range(len(jpgfile)):
        file_path=jpgfile[i]
        print("jpgfile[i]",i,jpgfile[i])
        hp_inference.predict(image_path=file_path)


if __name__ == '__main__':
    gtpath='label_1024'
    gt_dilate_path='label_1024_'
    
    for i in range(1,70):
        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_'+str(i)
        if os.path.exists(fatherpath):
            basepath = fatherpath+'/'+gtpath
            visualpath_statis = fatherpath+'/'+gt_dilate_path
            inference(basepath)
        else:
            pass

