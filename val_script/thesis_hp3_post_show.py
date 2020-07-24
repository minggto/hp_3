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
import skimage
from skimage import io,data,color,morphology,feature,measure
from skimage.measure import regionprops
from skimage.measure import label

import copy
import pandas as pd
import argparse




def find(path,label_path, pfix):
    # os.path.split（）返回文件的路径和文件名
    #os.path.splitext()将文件名和扩展名分开
    fname,fename=os.path.split(path)
    prename,postfix =os.path.splitext(fename)
    ffile = label_path+'/'+prename + pfix

    if os.path.exists(ffile):
        return ffile
    else:
        print("false",ffile)
        exit()

def inference(inference_path, label_path, gt_path):
    print("come in inference", inference_path)
    inference_files = glob(os.path.join(inference_path, '*.jpg'))
    jpgfile = inference_files

    pngfile = []
    for i in range(len(jpgfile)):
        pfile = find(jpgfile[i], label_path=label_path, pfix='.png')
        pngfile.append(pfile)

    gtfile = []
    for i in range(len(jpgfile)):
        pfile = find(jpgfile[i], label_path=gt_path, pfix='.png')
        gtfile.append(pfile)

    return jpgfile,pngfile,gtfile



def _post_processing(gpu_temp):
    # gpu_temp = np.array(gpu_temp[0, :, :], dtype=np.uint8)
    pos_mask = copy.copy(gpu_temp)
    pos_mask[pos_mask > 0] = 1
    # print("pos_mask.shape", pos_mask.shape)
    # 先进行图像闭运算，之后连通区域标记，删除其中小面积区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pos_mask_close = cv2.morphologyEx(pos_mask, cv2.MORPH_CLOSE, kernel)
    # print("pos_mask_close",pos_mask_close.shape,type(pos_mask_close))

    # 对连通区域标记后，将其中面积小于阈值的部分删除
    label_image = measure.label(pos_mask_close)  # 连通区域标记
    # dst=skimage.morphology.remove_small_objects(label_image, min_size=200, connectivity=1)
    dst = skimage.morphology.remove_small_objects(label_image, min_size=10, connectivity=1)

    '''对连通区域面积小于2000（已经没有面积小于200的区域了）的区域值置为1，只对连通区域面积大于2000的区域进行膨胀'''
    dst_temp = copy.copy(dst) * 0
    for region in measure.regionprops(dst):  # 循环得到每一个连通区域属性集
        # print("region.area",region.area)
        if region.area < 2000:
            for coordinates in region.coords:
                # print("dst_temp.shape",dst_temp.shape)
                dst_temp[coordinates[0], coordinates[1]] = 1

    dst[dst > 0] = 1
    pos_mask_close = np.asarray(dst - dst_temp, np.uint8)
    dila_kernel = np.ones((10, 10), np.uint8)
    pos_mask_close = cv2.dilate(pos_mask_close, dila_kernel, iterations=4)  # 膨胀
    pos_mask_close = pos_mask_close + dst_temp
    pos_mask_close = np.asarray(pos_mask_close, np.uint8)
    return pos_mask, pos_mask_close


def visural(imgfile_i,labfile_i,gtfile_i,visual_path):
# def visural(imgfile_i, labfile_i,  visual_path):
    fname, fename = os.path.split(imgfile_i)
    prename, postfix = os.path.splitext(fename)
    img_image = imageio.imread(imgfile_i)
    img_label = imageio.imread(labfile_i)
    img_gt = imageio.imread(gtfile_i)
    # img_image = cv2.imread(imgfile_i)
    # img_label = cv2.imread(labfile_i,cv2.IMREAD_GRAYSCALE)
    # img_gt = cv2.imread(gtfile_i,cv2.IMREAD_GRAYSCALE)
    # print(img_image.shape,img_label.shape)

    img_image = np.asarray(img_image, np.uint8)
    img_label = np.asarray(img_label, np.uint8)
    img_gt = np.asarray(img_gt, np.uint8)

    _, img_label = _post_processing(gpu_temp=img_label)
    img_label = np.asarray(img_label*255, np.uint8)

    # import pandas as pd
    # comp_pd = pd.DataFrame(img_label)
    # result = comp_pd.apply(pd.value_counts)
    # array_result = np.array(result.index)
    # result = np.sum(result, axis=1)
    # print("array_result", array_result)
    # print("result", result)

    temp_image = copy.copy(img_image)
    temp_image[:, :, 0][img_label == 0] = 0
    temp_image[:, :, 1][img_label == 0] = 0
    temp_image[:, :, 2][img_label == 0] = 0
    temp1_image = copy.copy(img_image)
    temp1_image[:, :, 0][img_label > 0] = 0
    temp1_image[:, :, 1][img_label > 0] = 0
    temp1_image[:, :, 2][img_label > 0] = 0
    blank_image = img_image * 0
    blank_image[:, :, 0][img_label == 255] = 255
    target1_mask = temp1_image + temp_image * 0.5 + blank_image * 0.5
    target1_mask = np.array(target1_mask,np.uint8)
    imageio.imwrite(os.path.join(visual_path, prename + '_pred_mask1.png'), target1_mask)
    imageio.imwrite(os.path.join(visual_path, prename + '_pred.png'), img_label)
    imageio.imwrite(os.path.join(visual_path, prename + '_pred_gt.png'), img_gt)

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
    target2_mask = np.array(target2_mask,np.uint8)
    imageio.imwrite(os.path.join(visual_path, prename + '_takegt_mask.png'), target2_mask)









parser = argparse.ArgumentParser()
parser.add_argument('--model_task', type=str, default="FCN_multi", help='model_task name.')
parser.add_argument('--checkpoint_i', type=int, default=639, help='checkpoint_i')
args = parser.parse_args()


'''
'/2_data/share/workspace/yym/HP/hp3_visural/pspnet_single1/split4096_vec_8_takegt/0579/takegt/'
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/split4096_vec_8/groundtruth_512/'
'''
if __name__ == '__main__':
    imagepath = 'image_512'
    test_basepath = "/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/"
    # dirs = os.listdir(test_basepath)

    model_task = args.model_task
    pred_basepath = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task
    takegt_basepath = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task


    checkpoint_i = args.checkpoint_i
    num = '%04d' % checkpoint_i

    vecs = ['split4096_vec_8','split4096_vec_9','split4096_vec_42','split4096_vec_48','split4096_vec_49']
    for k in range(len(vecs)):
        vec = vecs[k]
        basepath = test_basepath + vec + '/' + imagepath
        predpath = pred_basepath+ '/' + vec +  '/' + num
        takegtpath = takegt_basepath + '/' + vec + '_posttakegt' + '/' + num + '/takegt'

        imgfile, labfile, gtfile = inference(basepath, predpath, takegtpath)

        visual_path = takegt_basepath + '/' + vec + '_visural_post_target1_target2' + '/' + num
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)

        for i in range(len(imgfile)):
            visural(imgfile[i], labfile[i], gtfile[i], visual_path)
