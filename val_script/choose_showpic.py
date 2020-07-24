import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm
from glob import glob
from PIL import Image
import random

import cv2
from skimage.measure import regionprops
from skimage.measure import label
from skimage import io,data,color,morphology,feature,measure
import copy
import pandas as pd
import argparse

import shutil


def mymovefile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        # if not os.path.exists(fpath):
        #    os.makedirs(fpath)                #创建路径
        dstfile = os.path.join(dstpath, fname)
        #shutil.move(srcfile, dstfile)  # 移动文件
        shutil.copyfile(srcfile,dstfile)      #复制文件

        # print "move %s -> %s"%(srcfile,dstfile)


'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/split4096_vec_42/image_512/'
'/2_data/share/workspace/yym/HP/hp3_visural/FCN_multi/split4096_vec_42/0640/'
'/2_data/share/workspace/yym/HP/hp3_visural/FCN_multi/split4096_vec_42_visural_target1_target2/0640/'

'/2_data/share/workspace/yym/HP/hp3_visural/FCN_multi/split4096_vec_42_visural_post_target1_target2/0640/'

'/2_data/share/workspace/yym/HP/hp3_visural/show_data/fcn_multi/input_pred/'
'/2_data/share/workspace/yym/HP/hp3_visural/show_data/fcn_multi/takegt/'

''
if __name__ == '__main__':
    # vecs = ['split4096_vec_8', 'split4096_vec_9', 'split4096_vec_42', 'split4096_vec_48', 'split4096_vec_49']
    vecpic = [{'vec':'split4096_vec_49','pic':'splite_Top4096_left24576_Bottom8192_Right28672_scale40_3_crop_1_0'},
              {'vec':'split4096_vec_49','pic':'splite_Top4096_left24576_Bottom8192_Right28672_scale40_3_crop_1_1'},
              {'vec':'split4096_vec_42','pic':'splite_Top0_left12288_Bottom4096_Right16384_scale40_13_crop_1_0'},
              {'vec':'split4096_vec_42','pic':'splite_Top0_left12288_Bottom4096_Right16384_scale40_9_crop_1_0'},
              {'vec':'split4096_vec_42','pic':'splite_Top0_left12288_Bottom4096_Right16384_scale40_9_crop_1_1'},
              {'vec': 'split4096_vec_42', 'pic': 'splite_Top28672_left8192_Bottom32768_Right12288_scale40_3_crop_1_0'},
              {'vec':'split4096_vec_42','pic':'splite_Top28672_left8192_Bottom32768_Right12288_scale40_3_crop_0_1'},
              {'vec': 'split4096_vec_42', 'pic': 'splite_Top53248_left8192_Bottom57344_Right12288_scale40_4_crop_0_1'},
              {'vec': 'split4096_vec_42', 'pic': 'splite_Top53248_left8192_Bottom57344_Right12288_scale40_4_crop_0_1'},
              ]
    model_tasks = ['FCN_multi', 'FCN_single1', 'FCN_single2',
                   'Unet_multi','Unet_single1','Unet_single2',
                   'tiny_deeplabv3_multi', 'tiny_deeplabv3_single1', 'tiny_deeplabv3_single2',
                   'pspnet_multi', 'pspnet_single1', 'pspnet_single2',
                   'DANet_multi', 'DANet_single1', 'DANet_single2'
                   ]
    for i in range(len(model_tasks)):
        model_task = model_tasks[i]
        sourceFiles = []
        for j in range(len(vecpic)):
            sourceDir = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task + '/'+ vecpic[j]['vec'] + '_visural_post_target1_target2/0640/'
            if not os.path.exists(sourceDir):
                pass
            sourceFiles.append(sourceDir + vecpic[j]['pic'] + '_pred_mask1.png')
            sourceFiles.append(sourceDir + vecpic[j]['pic'] + '_takegt_mask.png')
        print(sourceFiles)
        targetDir = '/2_data/share/workspace/yym/HP/hp3_visural/show_data/'+ model_task +'/takegt'
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        for k in range(len(sourceFiles)):
            shutil.copy(sourceFiles[k], targetDir)

        sourceFiles = []
        for j in range(len(vecpic)):
            sourceDir = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/'+ vecpic[j]['vec'] +'/image_512/'
            sourceFiles.append(sourceDir + vecpic[j]['pic'] + '.jpg')
            sourceDir = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task + '/' + vecpic[j]['vec'] + '/0640/'
            sourceFiles.append(sourceDir + vecpic[j]['pic'] + '.png')
            sourceDir = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task + '/' + vecpic[j]['vec'] + '_visural_target1_target2/0640/'
            sourceFiles.append(sourceDir + vecpic[j]['pic'] + '_pred_mask1.png')

        targetDir = '/2_data/share/workspace/yym/HP/hp3_visural/show_data/'+ model_task +'/input_pred'
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        for k in range(len(sourceFiles)):
            shutil.copy(sourceFiles[k], targetDir)

