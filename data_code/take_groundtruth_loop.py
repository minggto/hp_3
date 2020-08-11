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



def calc_sigt(I,threshval):
    M,N = I.shape
    ulim = np.uint8(np.max(I))
    N1 = np.count_nonzero(I>threshval)
    N2 = np.count_nonzero(I<=threshval)
    # w1 = np.float64(N1)/(M*N)
    # w2 = np.float64(N2)/(M*N)
    w1 = np.float64(N1)/np.float64(N1+N2)
    w2 = np.float64(N2)/np.float64(N1+N2)
    #print N1,N2,w1,w2
    try:
        u1 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N1 for i in range(threshval+1,ulim))
        u2 = np.sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N2 for i in range(threshval+1))
        uT = u1*w1+u2*w2
        sigt = w1*w2*(u1-u2)**2
        #print u1,u2,uT,sigt
    except:
        return 0
    return sigt

#...........................................................................................

def get_threshold(I):
    max_sigt = 0
    opt_t = 0
    ulim = np.uint8(np.max(I))
    print(ulim)
    for t in range(ulim+1):
        sigt = calc_sigt(I,t)
        #print t, sigt
        if sigt > max_sigt:
            max_sigt = sigt
            opt_t = t
    print ('optimal high threshold: ',opt_t)
    return opt_t


def canny_threshold(img_temp):
    max_th = 127
    ala = img_temp * 0
    ala[img_temp < max_th] = 1
    vis_ala = ala * img_temp
    print("vis_ala:max", np.max(vis_ala))
    count_list = []
    img_list = []
    for i in range(1, max_th):
        count_img = img_temp * 0
        count_img[vis_ala == i] = 1
        count = np.sum(count_img)
        count_list.append(count)
        # for j in range(count):
        #     img_list.append(i)
    print('count_list', count_list)
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
    print("mu",mu)
    print("sigma",sigma)

    plt.subplot(subplot_num)
    #num_bins = 100 #直方图柱子的数量
    #n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5)
    # n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    num_bins = np.arange(0, np.max(x)+5, width)
    n, bins, patches = plt.hist(x, num_bins,  facecolor='blue')
    print("bins",bins,len(bins))
    print("n",n,len(n))
    print("sum(n)",sum(n))
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

    def predict(self, image_path, label_path, is_vis=True):
        label_temp = self._pre_processing(label_path) #self.ori_image 在label赋值后的图像是2维（4096，4096)
        label_temp = np.asarray(label_temp, dtype=np.uint8)
        label_temp=label_temp//255

        # groudtruth, img_temp, chull_image, gt_image = self.findgroudtruth_test(label_temp, image_path,visual_path=visualpath_statis)

        #先找label的groudtruth，之后再对label膨胀（以免groudtruth引入不必要的杂质）
        groudtruth,img_temp,chull_image,gt_image = self.findgroudtruth(label_temp,image_path,visual_path=visualpath_statis)

        self._visualization_statis(img_temp, groudtruth, image_path, visualpath_statis, chull_image, gt_image)

    def findgroudtruth_test(self, label_temp, image_path, visual_path):
        label = np.asarray(label_temp[0, :, :], dtype=np.uint8)
        image = imageio.imread(image_path)
        label_img = copy.copy(image)
        label_img[:, :, 0][label == 0] = 255
        label_img[:, :, 1][label == 0] = 255
        label_img[:, :, 2][label == 0] = 255

        img_temp = label_img[:, :, 0] * 0.8 + label_img[:, :, 1] * 0.2
        img_temp = np.asarray(img_temp, dtype=np.uint8)
        cv2.imshow('img_temp', img_temp)

        max_th = 127
        ala = img_temp*0
        ala[img_temp<max_th] = 1
        vis_ala = ala*img_temp
        cv2.imshow('vis_ala:127', vis_ala)

        # pos_mask = copy.copy(vis_ala)
        # pos_mask[vis_ala > 0] = 1
        # label_image = measure.label(pos_mask)  # 连通区域标记
        # big_objects = morphology.remove_small_objects(label_image, min_size=400, connectivity=1)
        #
        # for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
        #     if region.area > 400:
        #         print("region.area", region.area)
        # small_objects = label_image - big_objects

        ala_th = [0,10,20,30,40,50,60,70,80,90,100,110,120]
        # count_list=[]
        # for i in range(len(ala_th)):
        #     count_img = img_temp*0
        #     count_img[vis_ala > ala_th[i]] = 1
        #     count = np.sum(count_img)
        #     count_list.append(count)
        # cal_list=[]
        # for i in range(len(count_list)-1):
        #     cal = count_list[i+1] - count_list[i]
        #     cal_list.append(cal)
        # cal_list.append(count_list[-1])
        # print('cal_list',cal_list)

        # print("vis_ala:max",np.max(vis_ala))
        # count_list = []
        # img_list = []
        # for i in range(1,max_th):
        #     count_img = img_temp*0
        #     count_img[vis_ala == i] = 1
        #     count = np.sum(count_img)
        #     count_list.append(count)
        #     for j in range(count):
        #         img_list.append(i)
        # print('count_list',count_list)
        # mu_0, sigma_0 = data_analysis(img_list, 5, 111)
        # print("mu_0,sigma_0", mu_0, sigma_0)
        # plt.show()


        blan = copy.copy(label*255)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(blan, kernel, iterations=2)  # 膨胀
        erodion = cv2.erode(blan, kernel, iterations=2)  # 腐蚀
        annular = dilation - erodion
        # cv2.imshow('annular', annular)

        canny_th = canny_threshold(img_temp)
        edgs = cv2.Canny(img_temp, canny_th - 10, canny_th)
        # edgs = cv2.Canny(img_temp, 299, 300)
        # 检测canny边缘,得到二值图片
        #edgs = cv2.Canny(img_temp, 399, 400, 100)
        cv2.imshow('edgs1', edgs)

        edgs[annular == 255] = 0
        # cv2.imshow('edgs2', edgs)

        chull = skimage.morphology.convex_hull_object(edgs)
        chull = morphology.remove_small_objects(chull, min_size=10, connectivity=1)
        chull = np.asarray(chull, dtype=np.uint8)  # 这里要转换类型，下面才能膨胀
        # print("chull",type(chull),chull.shape)
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
        for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
            if region.area > 100:
                print("region.area", region.area)

        # label_image[label_image>0] = 255

        big_objects = np.array(big_objects, dtype=np.uint8)
        big_objects[big_objects > 0] = 255
        cv2.imshow('big_objects', big_objects)
        small_objects = gt - big_objects
        small_objects[small_objects>0] = 255
        cv2.imshow('small_objects', small_objects)

        gt = small_objects//255 * gt
        cv2.imshow('gt', gt)
        addgt = gt // 255 + chull_dilation // 255  # 这里uint8类型最大值255，如果超过值会越界变成不确定的数值，是一个大坑

        groudtruth = copy.copy(addgt)
        groudtruth[addgt < 2] = 0
        groudtruth[addgt >= 2] = 255
        groudtruth = np.asarray(groudtruth, dtype=np.uint8)
        cv2.imshow('groudtruth', groudtruth)

        temp_image = copy.copy(image)
        temp_image[:, :, 0][label == 0] = 0
        temp_image[:, :, 1][label == 0] = 0
        temp_image[:, :, 2][label == 0] = 0
        blank_image = copy.copy(temp_image)
        blank_image[:, :, 1][chull > 0] = 255
        chull_image = temp_image * 0.3 + blank_image * 0.7
        chull_image = np.asarray(chull_image, dtype=np.uint8)
        cv2.imshow('chull_image', chull_image[:,:,::-1])


        # blank_image = copy.copy(image) * 0
        # blank_image[:, :, 0][groudtruth > 0] = 255
        temp_image = copy.copy(image)
        temp_image[:, :, 0][label == 0] = 0
        temp_image[:, :, 1][label == 0] = 0
        temp_image[:, :, 2][label == 0] = 0
        blank_image = copy.copy(temp_image)
        blank_image[:, :, 0][groudtruth > 0] = 255
        gt_image = temp_image * 0.3 + blank_image * 0.7
        gt_image = np.asarray(gt_image, dtype=np.uint8)
        # cv2.imshow('blank_image', blank_image[:,:,::-1])
        # cv2.imshow('temp_image', temp_image[:,:,::-1])
        cv2.imshow('gt_image', gt_image[:,:,::-1])
        cv2.waitKey(0)

        return groudtruth, img_temp,chull_image,gt_image


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

        # temp_image = copy.copy(image)
        # temp_image[:, :, 0][label == 0] = 0
        # temp_image[:, :, 1][label == 0] = 0
        # temp_image[:, :, 2][label == 0] = 0
        # blank_image = copy.copy(temp_image)
        # blank_image[:, :, 1][chull > 0] = 255
        # chull_image = temp_image * 0.3 + blank_image * 0.7
        # chull_image = np.asarray(chull_image, dtype=np.uint8)


        # temp_image = copy.copy(image)
        # temp_image[:, :, 0][label == 0] = 0
        # temp_image[:, :, 1][label == 0] = 0
        # temp_image[:, :, 2][label == 0] = 0
        # blank_image = copy.copy(temp_image)
        # blank_image[:, :, 0][groudtruth > 0] = 255
        # gt_image = temp_image * 0.3 + blank_image * 0.7
        # gt_image = np.asarray(gt_image, dtype=np.uint8)

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

        '''
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        image_name = os.path.basename(image_path)
        imageio.imwrite(os.path.join(visual_path, image_name.split('.jpg')[0] + '.png'), groudtruth)
        imageio.imwrite(os.path.join(visual_path, image_name.split('.jpg')[0] + 'chull.png'), chull_dilation)
        imageio.imwrite(os.path.join(visual_path, image_name.split('.jpg')[0] + 'gt.png'), gt)
        '''

        if not os.path.exists(visual_path + '_canny_th'):
            os.makedirs(visual_path + '_canny_th')
        image_name = os.path.basename(image_path)
        string1=str(canny_th)
        writefile = visual_path + '_canny_th'+'/write_canny_th.txt'
        with open(writefile, 'a+') as f:
            f.write(string1)
            f.write(' ')
            f.write(image_name)
            f.write('\n')
        return groudtruth,img_temp,chull_image,gt_image
    


    def _visualization_statis(self, img_temp, groudtruth, image_path, visual_path,chull_image,gt_image):
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        if not os.path.exists(visual_path+'_img_temp'):
            os.makedirs(visual_path+'_img_temp')
        #写直接预测效果图
        #vis_path = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)

        img_temp = np.asarray(img_temp, dtype=np.uint8)
        imageio.imwrite(os.path.join(visual_path+'_img_temp', image_name.split('.jpg')[0] + '_img_temp.jpg'), img_temp)
        label = np.asarray(groudtruth, dtype=np.uint8)
        imageio.imwrite(os.path.join(visual_path, image_name.split('.jpg')[0] + '.png'), label)
  
        
        if not os.path.exists(visual_path+'_dilat_1'):
            os.makedirs(visual_path+'_dilat_1')
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(label,kernel,iterations = 10) #膨胀
        dilat_groudtruth = dilation
        imageio.imwrite(os.path.join(visual_path+'_dilat_1', image_name.split('.jpg')[0] + '.png'), dilat_groudtruth)
        
        '''
        if not os.path.exists(visual_path+'_dilat_2'):
            os.makedirs(visual_path+'_dilat_2')
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(label,kernel,iterations = 10) #膨胀
        dilat_groudtruth = dilation
        imageio.imwrite(os.path.join(visual_path+'_dilat_2', image_name.split('.jpg')[0] + '.png'), dilat_groudtruth)
        '''
        '''
        if not os.path.exists(visual_path+'_dilat_3'):
            os.makedirs(visual_path+'_dilat_3')
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(label,kernel,iterations = 5) #膨胀
        dilat_groudtruth = dilation
        imageio.imwrite(os.path.join(visual_path+'_dilat_3', image_name.split('.jpg')[0] + '.png'), dilat_groudtruth)
        '''

        if not os.path.exists(visual_path + '_visrualChull'):
            os.makedirs(visual_path + '_visrualChull')
        imageio.imwrite(os.path.join(visual_path + '_visrualChull', image_name.split('.jpg')[0] + '.png'), chull_image)

        if not os.path.exists(visual_path + '_visrualgt'):
            os.makedirs(visual_path + '_visrualgt')
        imageio.imwrite(os.path.join(visual_path + '_visrualgt', image_name.split('.jpg')[0] + '.png'), gt_image)



#filter=[".jpg",".JPG"] #设置过滤后的文件类型 当然可以设置多个类型
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


def inference(inference_path):

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
        hp_inference.predict(image_path=file_path, label_path=label_path, is_vis=True)


if __name__ == '__main__':
    imagepath='image_1024'
    labelpath = 'label_1024'
    # labelpath='label_1024_dilat'
    gtpath='groundtruth_1024'

    #basepath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/test_groudtruth/savefile_test_one/train_dataset/split4096_vec_1'+'/'+imagepath
    #visualpath_statis = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/test_groudtruth/savefile_test_one/train_dataset/split4096_vec_1'+'/'+gtpath

    # for i in range(0,70):
    for i in range(1,70):
        #fatherpath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/savefile/test_dataset/split4096_vec_'+str(i)
        #fatherpath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/test_groudtruth/savefile_41/test_dataset/split4096_vec_'+str(i)+'_gt'
        #fatherpath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/HpDataSet/train_dataset_1/split4096_vec_'+str(i)
        #fatherpath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/test_groudtruth/savefile_41/test_dataset_1_augmentation/split4096_vec_'+str(i)
        #fatherpath = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/test_groudtruth/savefile_41/test_dataset_1_new/split4096_vec_'+str(i)

        # fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/train_dataset_2_new/split4096_vec_' + str(i)
        fatherpath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/test_dataset_1_new/split4096_vec_' + str(i)
        if os.path.exists(fatherpath):
            basepath = fatherpath+'/'+imagepath
            visualpath_statis = fatherpath+'/'+gtpath
            inference(basepath)
        else:
            pass