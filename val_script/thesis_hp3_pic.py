import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm
from glob import glob
#import tensorflow as tf
from PIL import Image
import random

import cv2
from skimage.measure import regionprops
from skimage.measure import label
from skimage import io,data,color,morphology,feature,measure
import copy
import pandas as pd





def find(path,labelpath):
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

def inference(inference_path, labelpath, gtpath):
    print("come in inference", inference_path)
    inference_files = glob(os.path.join(inference_path, '*.jpg'))
    jpgfile = inference_files

    pngfile = []
    for i in range(len(jpgfile)):
        pfile, ppath = find(jpgfile[i], labelpath=labelpath)
        pngfile.append(pfile)

    gtfile = []
    for i in range(len(jpgfile)):
        pfile, ppath = find(jpgfile[i], labelpath=gtpath)
        gtfile.append(pfile)

    return jpgfile,pngfile,gtfile


def visural(imgfile_i,labfile_i,gtfile_i,visual_path):
    fname, fename = os.path.split(imgfile_i)
    prename, postfix = os.path.splitext(fename)
    img_image = imageio.imread(imgfile_i)
    img_label = imageio.imread(labfile_i)
    img_gt = imageio.imread(gtfile_i)
    img_image = np.asarray(img_image, np.uint8)
    img_label = np.asarray(img_label, np.uint8)
    img_gt = np.asarray(img_gt, np.uint8)

    temp_image = copy.copy(img_image)
    temp_image[:, :, 0][img_label == 0] = 0
    temp_image[:, :, 1][img_label == 0] = 0
    temp_image[:, :, 2][img_label == 0] = 0
    temp1_image = copy.copy(img_image)
    temp1_image[:, :, 0][img_label > 0] = 0
    temp1_image[:, :, 1][img_label > 0] = 0
    temp1_image[:, :, 2][img_label > 0] = 0
    blank_image = img_image * 0
    blank_image[:, :, 0][img_label > 0] = 255
    target1_mask = temp1_image + temp_image * 0.5 + blank_image * 0.5
    imageio.imwrite(os.path.join(visual_path, prename + '_target1_mask.jpg'), target1_mask)
    print(os.path.join(visual_path, prename + '_target1_mask.jpg'))
    imageio.imwrite(os.path.join(visual_path, prename + '_target1_extract.jpg'), temp_image)

    temp_label = copy.copy(img_label)
    edge = temp_label*0+1
    print(edge[5:-5,5:-5].shape)
    edge[5:-5,5:-5] = edge[5:-5,5:-5]*0
    temp_label_div_edge = temp_label*(1-edge)
    temp_label = copy.copy(temp_label_div_edge)
    dila_kernel = np.ones((10, 10), np.uint8)
    temp_label = cv2.dilate(temp_label, dila_kernel, iterations=1)  # 膨胀cv2.dilate
    # imageio.imwrite(os.path.join(visual_path, prename + '_temp_label_div_edge.jpg'), temp_label_div_edge)
    # imageio.imwrite(os.path.join(visual_path, prename + '_temp_label.jpg'), temp_label)
    temp_label = np.asarray(temp_label - temp_label_div_edge, np.uint8)
    temp_image = copy.copy(img_image)
    temp_image[:, :, 0][temp_label == 0] = 0
    temp_image[:, :, 1][temp_label == 0] = 0
    temp_image[:, :, 2][temp_label == 0] = 0
    temp1_image = copy.copy(img_image)
    temp1_image[:, :, 0][temp_label > 0] = 0
    temp1_image[:, :, 1][temp_label > 0] = 0
    temp1_image[:, :, 2][temp_label > 0] = 0
    blank_image = img_image * 0
    blank_image[:, :, 1][temp_label > 0] = 255
    target1_contour = temp1_image + temp_image * 0.2 + blank_image * 0.8
    imageio.imwrite(os.path.join(visual_path, prename + '_target1_contour.jpg'), target1_contour)

    # blank_image = img_image * 0
    # temp_image = copy.copy(blank_image)
    # temp_image[:, :, 0][img_gt > 0] = 0
    # temp_image[:, :, 1][img_gt > 64] = 255
    # temp_image[:, :, 2][img_gt > 0] = 0
    # imageio.imwrite(os.path.join(visual_path, image_file + '_target2_green.jpg'), temp_image)

    temp_image = copy.copy(img_image)
    temp_image[:, :, 0][img_gt == 0] = 0
    temp_image[:, :, 1][img_gt == 0] = 0
    temp_image[:, :, 2][img_gt == 0] = 0
    temp1_image = copy.copy(img_image)
    temp1_image[:, :, 0][img_gt > 0] = 0
    temp1_image[:, :, 1][img_gt > 0] = 0
    temp1_image[:, :, 2][img_gt > 0] = 0
    blank_image = img_image * 0
    blank_image[:, :, 0][img_gt > 0] = 255
    target2_mask = temp1_image + temp_image * 0.5 + blank_image * 0.5
    imageio.imwrite(os.path.join(visual_path, prename + '_target2_mask.jpg'), target2_mask)


    temp_image = copy.copy(target1_contour)
    temp_image[:, :, 0][img_gt == 0] = 0
    temp_image[:, :, 1][img_gt == 0] = 0
    temp_image[:, :, 2][img_gt == 0] = 0
    temp1_image = copy.copy(target1_contour)
    temp1_image[:, :, 0][img_gt > 0] = 0
    temp1_image[:, :, 1][img_gt > 0] = 0
    temp1_image[:, :, 2][img_gt > 0] = 0
    blank_image = img_image * 0
    blank_image[:, :, 0][img_gt > 0] = 255
    target1_contour_target2_mask = temp1_image + temp_image * 0.5 + blank_image * 0.5
    imageio.imwrite(os.path.join(visual_path, prename + '_target1_contour_target2_mask.jpg'), target1_contour_target2_mask)




basepath = '/2_data/share/workspace/yym/HP/hp3_visural/ShowData/Polygon and Targets/'
imagepath = basepath+'images'
labelpath = basepath+'target1'
gtpath = basepath+'target2'
imgfile,labfile,gtfile = inference(imagepath, labelpath, gtpath)

visual_path = basepath+'visural_target1_target2'
if not os.path.exists(visual_path):
    os.makedirs(visual_path)

if __name__ == '__main__':
    for i in range(len(imgfile)):
        visural(imgfile[i], labfile[i], gtfile[i],visual_path)






