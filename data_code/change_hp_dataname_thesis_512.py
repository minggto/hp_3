
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


def load_pkl(file_name):
    with open(file_name, "rb") as f:
        data_pkl = pickle.load(f)
        if data_pkl.__contains__('x_train'):
            x_filenames = data_pkl['x_train']
            y_filenames = data_pkl['y_train']
            y1_filenames = data_pkl['z_train']
        elif data_pkl.__contains__('x_valid'):
            x_filenames = data_pkl['x_valid']
            y_filenames = data_pkl['y_valid']
            y1_filenames = data_pkl['z_valid']

    for i in range(len(y_filenames)):
        y_filenames[i] = y_filenames[i].replace("groundtruth_512_dilat30","groundtruth_512_dilat9")
    print(x_filenames[0])
    print(y_filenames[0])
    print(y1_filenames[0])
    print(len(x_filenames))

    return x_filenames,y_filenames,y1_filenames




if __name__ == '__main__':

    pkl_path='/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512_fix/'
    train_file_name = pkl_path+'dataname_train.pkl'
    x_train_filenames,y_train_filenames,y1_train_filenames = load_pkl(train_file_name)
    valid_file_name = pkl_path + 'dataname_valid.pkl'
    x_valid_filenames, y_valid_filenames, y1_valid_filenames = load_pkl(valid_file_name)

    print("Saving the data numpy pickle to the disk..")
    new_pkl_path = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/dataname_pkl_512/'
    if not os.path.exists(new_pkl_path):
        os.makedirs(new_pkl_path)

    new_file_name = new_pkl_path+'dataname_train.pkl'
    with open(new_file_name, 'wb')as f:
        pickle.dump({'x_train': x_train_filenames,
                     'y_train': y_train_filenames,
                     'z_train': y1_train_filenames,
                     }, f)


    new_file_name = new_pkl_path+'dataname_valid.pkl'
    with open(new_file_name, 'wb')as f:
        pickle.dump({'x_valid': x_valid_filenames,
                     'y_valid': y_valid_filenames,
                     'z_valid': y1_valid_filenames,
                     }, f)

    # file_name = pkl_path+'dataname_test.pkl'
    # with open(file_name, 'wb')as f:
    #     pickle.dump({'x_test': x_test_filenames,
    #                  'y_test': y_test_filenames,
    #                  'z_test': y1_test_filenames,
    #                  }, f)

    print("DATA NUMPY PICKLE saved successfully..")
    

