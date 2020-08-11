import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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



class HpInference:
    def __init__(self):       
        self.ori_image = None
        self.image_shape = None


    def predict(self, label_path, image_path, new_image_path, new_label_path):
        label = imageio.imread(label_path)
        #对连通区域标记
        label=label//255
        sum_area=np.sum(label)
        if sum_area > 100:
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
        old_temppath = label_path.split(labelpath)[0]+labelpath_dilat+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+labelpath_dilat
        if not os.path.exists(new_temp_path):
            os.makedirs(new_temp_path)
        shutil.copyfile(old_temppath,new_temp_path+'/'+temp_name)

        temp_name = label_name
        old_temppath = label_path.split(labelpath)[0]+groundtruthpath_erode+'/'+temp_name
        new_temp_path = new_label_path.split(labelpath)[0]+groundtruthpath_erode
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



def inference(old_imagepath,old_labelpath,new_imagepath,new_labelpath):
    inference_path = old_imagepath
    inference_files = glob(os.path.join(inference_path, '*.jpg'))
    jpgfile=inference_files

    pngfile=[]
    for i in range(len(jpgfile)):
        pfile,ppath=find(jpgfile[i])
        pngfile.append(pfile)

    print("inference_files",jpgfile[0],pngfile[0])
    
    hp_inference = HpInference()
    for i in range(len(jpgfile)):
        file_path=jpgfile[i]
        print("jpgfile[i]",i,jpgfile[i])
        label_path=pngfile[i]
        hp_inference.predict(image_path=file_path, label_path=label_path, new_image_path=new_imagepath, new_label_path=new_labelpath)


imagepath='image_512'
labelpath='label_512'
groundtruthpath='groundtruth_512'

labelpath_dilat='label_512_dilat'
groundtruthpath_erode='groundtruth_512_erode'



if __name__ == '__main__':

    for i in range(1,70):

        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512/split4096_vec_'+str(i)
        new_fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_1_new_512_filter/split4096_vec_'+str(i)

        if os.path.exists(fatherpath):
            old_imagepath = fatherpath+'/'+imagepath
            old_labelpath = fatherpath+'/'+labelpath
            new_imagepath = new_fatherpath+'/'+imagepath
            new_labelpath = new_fatherpath+'/'+labelpath
            inference(old_imagepath,old_labelpath,new_imagepath,new_labelpath)
        else:
            pass
