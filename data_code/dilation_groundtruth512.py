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
        self._visualization(groudtruth, image_path, visual_path=vispath)


    def _visualization(self, groudtruth, image_path, visual_path):
        image_name = os.path.basename(image_path)
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(groudtruth,kernel,iterations = 3) #膨胀
        dilat_groudtruth = dilation
        imageio.imwrite(os.path.join(visual_path, image_name.split('.png')[0] + '.png'), dilat_groudtruth)



def inference(inference_path):
    inference_files = glob(os.path.join(inference_path, '*.png'))

    hp_inference = HpInference()
    for i in range(len(inference_files)):
        file_path=inference_files[i]
        print("file[i]",i,inference_files[i])
        hp_inference.predict(image_path=file_path)


if __name__ == '__main__':
    # basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter'
    # labelpath_gt = 'groundtruth_512'
    # build_gtdilat = 'groundtruth_512_dilat'

    basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512'
    labelpath_gt = 'groundtruth_512'
    build_gtdilat = 'groundtruth_512_dilat9'
    
    for i in range(1,70):
        fatherpath = basepath + '/split4096_vec_'+str(i)
        if os.path.exists(fatherpath):
            dirpath = fatherpath+'/'+labelpath_gt
            vispath = fatherpath+'/'+build_gtdilat
            inference(dirpath)
        else:
            pass

