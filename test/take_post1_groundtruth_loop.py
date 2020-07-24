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
import argparse




#...........................................................................................




def canny_threshold(img_temp):
    max_th = 127
    ala = img_temp * 0
    ala[img_temp < max_th] = 1
    vis_ala = ala * img_temp
    # print("vis_ala:max", np.max(vis_ala))
    count_list = []
    img_list = []
    for i in range(1, max_th):
        count_img = img_temp * 0
        count_img[vis_ala == i] = 1
        count = np.sum(count_img)
        count_list.append(count)
        # for j in range(count):
        #     img_list.append(i)
    # print('count_list', count_list)
    # mu_0, sigma_0 = data_analysis(img_list, 5, 111)
    # print("mu_0,sigma_0", mu_0, sigma_0)

    # r = 1.0*np.sum(count_list[70:-1])/np.sum(count_list)
    # canny_th = (1-r)*200+210
    # r1 = 1.0*np.sum(count_list[0:10])/np.sum(count_list)
    # r2 = 1.0 * np.sum(count_list[0:20]) / np.sum(count_list)
    r=[]
    sum_count = np.sum(count_list)
    for k in range(0,120,10):
        rtemp = 1.0 * np.sum(count_list[0:k]) / sum_count
        r.append(rtemp)
    canny_th = 160
    for i in range(len(r)):
        canny_th = canny_th + r[i]*20
    print("canny_th",canny_th)
    return canny_th

def data_analysis(x,  width, subplot_num):
    # 数据
    # mu = 100  # 均值
    # sigma = 20  # 方差
    # # 2000个数据
    # x = mu + sigma * np.random.randn(2000)
    # print("x",x)
    # # 画图 bins:条形的个数， normed：是否标准化
    # plt.hist(x=x, bins=20)
    # # 展示
    # plt.show()

    x = np.array(x)
    mu =np.mean(x) #计算均值
    sigma =np.std(x)
    # print("mu",mu)
    # print("sigma",sigma)

    plt.subplot(subplot_num)
    #num_bins = 100 #直方图柱子的数量
    #n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5)
    # n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    num_bins = np.arange(0, np.max(x)+5, width)
    n, bins, patches = plt.hist(x, num_bins,  facecolor='blue')
    # print("bins",bins,len(bins))
    # print("n",n,len(n))
    # print("sum(n)",sum(n))
    #直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
    # y = mlab.normpdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y
    #
    # plt.plot(bins, y, 'r--') #绘制y的曲线
    # plt.xlabel('sepal-length') #绘制x轴
    # plt.ylabel('Probability') #绘制y轴
    # plt.title(r'Histogram : $\mu={}$,$\sigma={}$'.format('%.3f'%mu,'%.3f'%sigma))#中文标题 u'xxx'
    # plt.subplots_adjust(left=0.15)#左边距
    # #plt.show()
    return mu,sigma

class HpInference:
    def __init__(self):
        
        self.ori_image = None
        self.image_shape = None
        
    def _pre_processing(self, image_path):
        image = imageio.imread(image_path)       
        self.image_shape = list(image.shape)
        return image.reshape([1] + list(image.shape))

    def _post_processing(self, gpu_temp):
        gpu_temp = np.array(gpu_temp[0, :, :], dtype=np.uint8)

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

    def _post1_processing(self, gpu_temp):
        gpu_temp = np.array(gpu_temp[0, :, :], dtype=np.uint8)

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

        dst[dst > 0] = 1
        pos_mask_close = np.asarray(dst, np.uint8)

        dila_kernel = np.ones((10, 10), np.uint8)
        pos_mask_close = cv2.dilate(pos_mask_close, dila_kernel, iterations=4)  # 膨胀

        pos_mask_close = np.asarray(pos_mask_close, np.uint8)
        return pos_mask, pos_mask_close


    def predict(self, image_path, label_path, visual_path, is_vis=True):
        label_temp = self._pre_processing(label_path) #self.ori_image 在label赋值后的图像是2维（4096，4096)
        label_temp = np.asarray(label_temp, dtype=np.uint8)
        # label_temp=label_temp//255
        label_temp[label_temp >0] = 1
        # print("label_temp.shape", label_temp.shape)
        pos_mask, pos_mask_close = self._post1_processing(label_temp)
        pos_mask_close = np.asarray(pos_mask_close, dtype=np.uint8).reshape([1] + list(pos_mask_close.shape))

        groudtruth,img_temp,chull_image,gt_image = self.findgroudtruth(pos_mask_close,image_path,visual_path)

        self._visualization_statis(img_temp, groudtruth, image_path, visual_path, chull_image, gt_image)




    def findgroudtruth_small(self,img_temp,annular):
        canny_th = canny_threshold(img_temp)
        edgs = cv2.Canny(img_temp, canny_th - 10, canny_th)
        edgs[annular == 255] = 0
        # cv2.imshow('edgs2', edgs)

        chull = skimage.morphology.convex_hull_object(edgs)
        chull = morphology.remove_small_objects(chull, min_size=10, connectivity=1)
        chull = np.asarray(chull, dtype=np.uint8)  # 这里要转换类型，下面才能膨胀
        chull[chull > 0] = 255  # 这里要转换为255，才能显示出来，否则值可能是1
        chull[chull <= 0] = 0
        # cv2.imshow('chull2', chull)

        kernel = np.ones((3, 3), np.uint8)
        chull_dilation = cv2.dilate(chull, kernel, iterations=1)  # 膨胀
        chull_dilation[chull_dilation > 0] = 255
        chull_dilation[chull_dilation <= 0] = 0
        # cv2.imshow('chull_dilation2', chull_dilation)

        gt = copy.copy(img_temp)
        th = 127
        # th = 159
        gt[img_temp < th] = 255
        gt[img_temp >= th] = 0

        pos_mask = copy.copy(gt)
        pos_mask[gt > 0] = 1
        label_image = measure.label(pos_mask)  # 连通区域标记
        big_objects = morphology.remove_small_objects(label_image, min_size=100, connectivity=1)
        # for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
        #     if region.area > 100:
        #         print("region.area", region.area)

        big_objects = np.array(big_objects, dtype=np.uint8)
        big_objects[big_objects > 0] = 255
        # cv2.imshow('big_objects', big_objects)
        small_objects = gt - big_objects
        small_objects[small_objects > 0] = 255
        # cv2.imshow('small_objects', small_objects)

        gt = small_objects // 255 * gt
        # cv2.imshow('gt', gt)
        addgt = gt // 255 + chull_dilation // 255  # 这里uint8类型最大值255，如果超过值会越界变成不确定的数值，是一个大坑

        groudtruth = copy.copy(addgt)
        groudtruth[addgt < 2] = 0
        groudtruth[addgt >= 2] = 255
        groudtruth = np.asarray(groudtruth, dtype=np.uint8)
        return groudtruth,canny_th,chull_dilation

    def findgroudtruth_old(self,img_temp,annular):
        edgs = cv2.Canny(img_temp, 399, 400)
        edgs[annular == 255] = 0

        chull = skimage.morphology.convex_hull_object(edgs)
        chull = np.asarray(chull, dtype=np.uint8)  # 这里要转换类型，下面才能膨胀
        chull[chull > 0] = 255  # 这里要转换为255，才能显示出来，否则值可能是1
        chull[chull <= 0] = 0

        kernel = np.ones((5, 5), np.uint8)
        chull_dilation = cv2.dilate(chull, kernel, iterations=1)  # 膨胀
        chull_dilation[chull_dilation > 0] = 255
        chull_dilation[chull_dilation <= 0] = 0

        # th=127
        th = 159
        gt = copy.copy(img_temp)
        gt[img_temp < th] = 255
        gt[img_temp >= th] = 0

        addgt = gt // 255 + chull_dilation // 255  # 这里uint8类型最大值255，如果超过值会越界变成不确定的数值，是一个大坑
        groudtruth = copy.copy(addgt)
        groudtruth[addgt < 2] = 0
        groudtruth[addgt >= 2] = 255
        groudtruth = np.asarray(groudtruth, dtype=np.uint8)
        return groudtruth,chull_dilation


    def findgroudtruth(self,label_temp,image_path,visual_path):
        label = np.asarray(label_temp[0, :, :], dtype=np.uint8)
        image = imageio.imread(image_path)
        label_img = copy.copy(image)
        label_img[:, :, 0][label == 0] = 255
        label_img[:, :, 1][label == 0] = 255
        label_img[:, :, 2][label == 0] = 255

        img_temp = label_img[:, :, 0] * 0.8 + label_img[:, :, 1] * 0.2
        img_temp = np.asarray(img_temp, dtype=np.uint8)

        blan = copy.copy(label * 255)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(blan, kernel, iterations=2)  # 膨胀
        erodion = cv2.erode(blan, kernel, iterations=2)  # 腐蚀
        annular = dilation - erodion

        groudtruth_old,chull_dilation_old = self.findgroudtruth_old(img_temp, annular)
        groudtruth_small,canny_th,chull_dilation_small = self.findgroudtruth_small(img_temp, annular)

        groudtruth = copy.copy(groudtruth_old)
        groudtruth[groudtruth_small==255] = 255


        temp_image = copy.copy(image)
        temp_image[:, :, 0][label == 0] = 0
        temp_image[:, :, 1][label == 0] = 0
        temp_image[:, :, 2][label == 0] = 0
        blank_image = copy.copy(temp_image)
        blank_image[:, :, 0][chull_dilation_old  > 0] = 255
        blank_image[:, :, 1][chull_dilation_small > 0] = 255
        chull_image = temp_image * 0.3 + blank_image * 0.7
        chull_image = np.asarray(chull_image, dtype=np.uint8)

        temp_image = copy.copy(image)
        temp_image[:, :, 0][label == 0] = 0
        temp_image[:, :, 1][label == 0] = 0
        temp_image[:, :, 2][label == 0] = 0
        blank_image = copy.copy(temp_image)
        blank_image[:, :, 0][groudtruth_old > 0] = 255
        blank_image[:, :, 1][groudtruth_small > 0] = 255
        gt_image = temp_image * 0.3 + blank_image * 0.7
        gt_image = np.asarray(gt_image, dtype=np.uint8)


        if not os.path.exists(visual_path + '/canny_th'):
            os.makedirs(visual_path + '/canny_th')
        image_name = os.path.basename(image_path)
        string1=str(canny_th)
        writefile = visual_path + '/canny_th'+'/write_canny_th.txt'
        with open(writefile, 'a+') as f:
            f.write(string1)
            f.write(' ')
            f.write(image_name)
            f.write('\n')
        return groudtruth,img_temp,chull_image,gt_image
    


    def _visualization_statis(self, img_temp, groudtruth, image_path, visual_path,chull_image,gt_image):
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        if not os.path.exists(visual_path+'/img_temp'):
            os.makedirs(visual_path+'/img_temp')
        if not os.path.exists(visual_path+'/takegt'):
            os.makedirs(visual_path+'/takegt')

        #vis_path = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)

        img_temp = np.asarray(img_temp, dtype=np.uint8)
        imageio.imwrite(os.path.join(visual_path+'/img_temp', image_name.split('.jpg')[0] + '_img_temp.jpg'), img_temp)
        label = np.asarray(groudtruth, dtype=np.uint8)
        imageio.imwrite(os.path.join(visual_path+'/takegt', image_name.split('.jpg')[0] + '.png'), label)
  
        
        # if not os.path.exists(visual_path+'/dilat9'):
        #     os.makedirs(visual_path+'/dilat9')
        # kernel = np.ones((3,3),np.uint8)
        # dilation = cv2.dilate(label,kernel,iterations = 3) #膨胀
        # dilat_groudtruth = dilation
        # imageio.imwrite(os.path.join(visual_path+'/dilat9', image_name.split('.jpg')[0] + '.png'), dilat_groudtruth)
        #
        #
        #
        # if not os.path.exists(visual_path + '/visrualChull'):
        #     os.makedirs(visual_path + '/visrualChull')
        # imageio.imwrite(os.path.join(visual_path + '/visrualChull', image_name.split('.jpg')[0] + '.png'), chull_image)
        #
        # if not os.path.exists(visual_path + '/visrualgt'):
        #     os.makedirs(visual_path + '/visrualgt')
        # imageio.imwrite(os.path.join(visual_path + '/visrualgt', image_name.split('.jpg')[0] + '.png'), gt_image)





def inference(basepath,basepath_pred,basepath_takegt):

    inference_files = glob(os.path.join(basepath, '*.jpg'))
    jpgfile=inference_files

    pngfile=[]
    for i in range(len(jpgfile)):
        fname, fename = os.path.split(jpgfile[i])
        prename, postfix = os.path.splitext(fename)
        pfile = basepath_pred + '/' + prename + '.png'
        pngfile.append(pfile)

    # print("inference_files",jpgfile[0],pngfile[0])

    hp_inference = HpInference()
    for i in range(len(jpgfile)):
        file_path=jpgfile[i]
        # print("jpgfile[i]",i,jpgfile[i])
        label_path=pngfile[i]
        hp_inference.predict(image_path=file_path, label_path=label_path, visual_path = basepath_takegt, is_vis=True)



parser = argparse.ArgumentParser()
parser.add_argument('--model_task', type=str, default="FCN_multi", help='model_task name.')
parser.add_argument('--checkpoint_i', type=int, default=639, help='checkpoint_i')
args = parser.parse_args()




if __name__ == '__main__':
    imagepath='image_512'
    # labelpath = 'label_512'
    # gtpath='groundtruth_1024'

    for i in range(1,70):
        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/split4096_vec_' + str(i)
        fatherpath_pred = '/2_data/share/workspace/yym/HP/hp3_visural/' + args.model_task + '/split4096_vec_' + str(i)
        fatherpath_takegt = fatherpath_pred + '_post1_takegt'

        if not os.path.exists(fatherpath):
            pass
        else:
            # for checkpoint_i in range(639, 641, 2):
            #     num = '%04d' % checkpoint_i
            #     basepath = fatherpath + '/' + imagepath
            #     basepath_pred = fatherpath_pred+ '/' + num
            #     basepath_takegt = fatherpath_takegt + '/' + num
            #
            #     inference(basepath,basepath_pred,basepath_takegt)

            checkpoint_i = args.checkpoint_i
            num = '%04d' % checkpoint_i
            basepath = fatherpath + '/' + imagepath
            basepath_pred = fatherpath_pred+ '/' + num
            basepath_takegt = fatherpath_takegt + '/' + num
            print("take_post1_groundtruth_loop.py",args.model_task,num,'split4096_vec_{}'.format(i))

            inference(basepath,basepath_pred,basepath_takegt)

