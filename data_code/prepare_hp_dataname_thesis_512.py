
import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from PIL import Image
import random



def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def save_imgs_to_disk(path, arr, file_names):
    for i, img in tqdm(enumerate(arr)):
        imageio.imwrite(path + file_names[i], img, 'PNG-PIL')


def save_numpy_to_disk(path, arr):
    np.save(path, arr)


def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())



filter=[".jpg",".JPG"] #设置过滤后的文件类型 当然可以设置多个类型
imagepath='image_512'


def all_path(dirname):
    result = []#所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):
        #print("1:",maindir) #当前主目录
        #print("2:",subdir) #当前主目录下的所有目录
        #print("3:",file_name_list)  #当前主目录下的所有文件
        #print("maindir",maindir.split('/',-1)[-1])
        if maindir.split('/',-1)[-1]==imagepath:
            #print(maindir)
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                #result.append(apath)            
                ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
                if ext in filter:
                    result.append(apath)
            
    return result


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




def all_filenames(basepath,labelpath_gt,labelpath,random_flag):
    jpgfile = all_path(basepath)
    if random_flag:
        random.shuffle(jpgfile)  # shuffle trainset filenames

    pngfile = []
    pngfile_1 = []

    train_num = len(jpgfile)
    print("train_num", train_num)
    all_filenames = jpgfile

    for i in range(len(jpgfile)):
        pfile, ppath = find(jpgfile[i], labelpath=labelpath_gt)
        pngfile.append(pfile)
    all_filenames_png = pngfile

    x_train_filenames = all_filenames
    y_train_filenames = all_filenames_png

    for i in range(len(jpgfile)):
        pfile1, ppath1 = find(jpgfile[i], labelpath=labelpath)
        pngfile_1.append(pfile1)

    y1_train_filenames = pngfile_1
    return x_train_filenames,y_train_filenames,y1_train_filenames




if __name__ == '__main__':
    basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512'
    labelpath = 'label_512'
    labelpath_gt = 'groundtruth_512_dilat'
    random_flag = True
    x_train_filenames,y_train_filenames,y1_train_filenames = all_filenames(basepath, labelpath_gt, labelpath, random_flag)


    basepath='/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset_512'
    labelpath = 'label_512'
    labelpath_gt = 'groundtruth_512_dilat'
    random = False
    x_valid_filenames,y_valid_filenames,y1_valid_filenames = all_filenames(basepath, labelpath_gt, labelpath, random)


    basepath='/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512'
    labelpath = 'label_512'
    labelpath_gt = 'groundtruth_512'
    random = False
    x_test_filenames,y_test_filenames,y1_test_filenames = all_filenames(basepath, labelpath_gt, labelpath, random)


    print(len(x_train_filenames),len(x_test_filenames),len(x_valid_filenames))
    print(len(y_train_filenames), len(y_test_filenames), len(y_valid_filenames))


    
    print("Saving the data numpy pickle to the disk..")
    pkl_path='/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512_fix/'
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)


    # SAVE ALL the data with one pickle
    file_name = pkl_path+'dataname_train.pkl'
    with open(file_name, 'wb')as f:
        pickle.dump({'x_train': x_train_filenames,
                     'y_train': y_train_filenames,
                     'z_train': y1_train_filenames,
                     }, f)

    file_name = pkl_path+'dataname_test.pkl'
    with open(file_name, 'wb')as f:
        pickle.dump({'x_test': x_test_filenames,
                     'y_test': y_test_filenames,
                     'z_test': y1_test_filenames,
                     }, f)

    file_name = pkl_path+'dataname_valid.pkl'
    with open(file_name, 'wb')as f:
        pickle.dump({'x_valid': x_valid_filenames,
                     'y_valid': y_valid_filenames,
                     'z_valid': y1_valid_filenames,
                     }, f)

    print("DATA NUMPY PICKLE saved successfully..")
    

