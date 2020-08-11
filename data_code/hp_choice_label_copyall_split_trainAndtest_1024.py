import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
from skimage import io,data,color,morphology,feature,measure
import shutil
import random


class HpInference:
    def __init__(self):       
        self.ori_image = None
        self.image_shape = None


    def predict(self, label_path, image_path, new_image_path, new_label_path):
        '''
        label = imageio.imread(label_path)
        #对连通区域标记
        label=label//255
        sum_area=np.sum(label)
        if sum_area > 2000: 
            self.copyfile(label_path, image_path, new_image_path, new_label_path)
        '''
        self.copyfile(label_path, image_path, new_image_path, new_label_path)

    
    def copyfile(self, label_path, image_path, new_image_path,new_label_path):
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        if not os.path.exists(new_image_path):
            os.makedirs(new_image_path)
        if not os.path.exists(new_label_path):
            os.makedirs(new_label_path)

        shutil.copyfile(label_path,new_label_path+'/'+label_name)
        shutil.copyfile(image_path,new_image_path+'/'+image_name)
        #imageio.imwrite(os.path.join(visual_path+'dilat_2', image_name.split('.png')[0] + '.png'), dilat_groudtruth)

        # groundtruth_name = label_name
        # old_groundtruthpath = label_path.split(labelpath)[0]+groundtruthpath+'/'+groundtruth_name
        # new_groundtruth_path = new_label_path.split(labelpath)[0]+groundtruthpath
        # if not os.path.exists(new_groundtruth_path):
        #     os.makedirs(new_groundtruth_path)
        # shutil.copyfile(old_groundtruthpath,new_groundtruth_path+'/'+groundtruth_name)


        temp_name = label_name
        old_temppath = label_path.split(labelpath)[0]+groundtruthpath+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+groundtruthpath
        if not os.path.exists(new_temp_path):
            os.makedirs(new_temp_path)
        shutil.copyfile(old_temppath,new_temp_path+'/'+temp_name)

        temp_name = label_name
        old_temppath = label_path.split(labelpath)[0]+groundtruthpath_dilat+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+groundtruthpath_dilat
        if not os.path.exists(new_temp_path):
            os.makedirs(new_temp_path)
        shutil.copyfile(old_temppath,new_temp_path+'/'+temp_name)


        temp_name = label_name
        old_temppath = label_path.split(labelpath)[0]+labelpath2+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+labelpath2
        if not os.path.exists(new_temp_path):
            os.makedirs(new_temp_path)
        shutil.copyfile(old_temppath,new_temp_path+'/'+temp_name)

        temp_name = label_name
        old_temppath = label_path.split(labelpath)[0]+groundtruthpath2+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+groundtruthpath2
        if not os.path.exists(new_temp_path):
            os.makedirs(new_temp_path)
        shutil.copyfile(old_temppath,new_temp_path+'/'+temp_name)



def find(path):
    result = []#所有的文件
    #os.path.splitext()将文件名和扩展名分开
    #os.path.split（）返回文件的路径和文件名
    fname,fename=os.path.split(path)
    prename,postfix =os.path.splitext(fename)
    #print("fname",fname)
    fpath=fname.split(imagepath,-1)[0]
    #print("split",fpath)
    ffile=fpath+labelpath+'/'+prename+'.png'
    #print(ffile)
    if os.path.exists(ffile):
        #print("true")
        return ffile,fpath
    else:
        print("false")
        return -1



def inference(old_imagepath,old_labelpath,train_imagepath,train_labelpath,test_imagepath,test_labelpath):
    inference_path = old_imagepath
    inference_files = glob(os.path.join(inference_path, '*.jpg'))
    jpgfile=inference_files
    random.shuffle(jpgfile)

    pngfile=[]
    for i in range(len(jpgfile)):
        pfile,ppath=find(jpgfile[i])
        pngfile.append(pfile)

    print("inference_files",jpgfile[0],pngfile[0])
    
    hp_inference = HpInference()
    length=len(jpgfile)*1//10
    for i in range(0,length):
        file_path=jpgfile[i]
        print("jpgfile[i]",i,jpgfile[i])
        label_path=pngfile[i]
        hp_inference.predict(image_path=file_path, label_path=label_path, new_image_path=test_imagepath, new_label_path=test_labelpath)

    for i in range(length,len(jpgfile)):
        file_path=jpgfile[i]
        print("jpgfile[i]",i,jpgfile[i])
        label_path=pngfile[i]
        hp_inference.predict(image_path=file_path, label_path=label_path, new_image_path=train_imagepath, new_label_path=train_labelpath)




imagepath='image_1024'
labelpath='label_1024'
groundtruthpath='groundtruth_1024'
groundtruthpath_dilat='groundtruth_1024_dilat_2'

labelpath2='label_1024_dilat'
groundtruthpath2='groundtruth_1024_erode'


if __name__ == '__main__':
    for i in range(0,70):

        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_1_new/split4096_vec_' + str(i)

        train_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset/split4096_vec_' + str(i)
        test_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset/split4096_vec_' + str(i)

        if os.path.exists(fatherpath):
            old_imagepath = fatherpath+'/'+imagepath
            old_labelpath = fatherpath+'/'+labelpath
            # old_labeldilatpath= fatherpath + '/' + labeldilatpath
            train_imagepath = train_fatherpath+'/'+imagepath
            train_labelpath = train_fatherpath+'/'+labelpath
            # train_labeldilatpath = train_fatherpath + '/' + labeldilatpath
            test_imagepath = test_fatherpath+'/'+imagepath
            test_labelpath = test_fatherpath+'/'+labelpath
            # test_labeldilatpath = test_fatherpath + '/' + labeldilatpath
            inference(old_imagepath,old_labelpath,train_imagepath,train_labelpath,test_imagepath,test_labelpath)
        else:
            pass
