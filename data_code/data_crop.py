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
import pickle
import math



def crop_jpg_png(image_files, basepath, newbasepath, imagepath, newimagepath):
	newimage_files = []
	crop_size = 512
	for i in range(len(image_files)):
		img = imageio.imread(image_files[i])
		img = img.astype(np.uint8)
		# print(img.shape)
		m0 = math.ceil(img.shape[0] / crop_size)
		m1 = math.ceil(img.shape[1] / crop_size)
		# print("m0,m1", m0,m1)
		if m0>1 or m1>1:
			for j0 in range(m0):
				for j1 in range(m1):
					# print("j0,j1", j0, j1, (j0+1)*crop_size, img.shape[0])
					h = np.min([(j0+1)*crop_size, img.shape[0]])
					w = np.min([(j1 + 1) * crop_size, img.shape[1]])
					# print("h,w",h,w)
					filepath, filename = os.path.split(image_files[i])
					filename, extension = os.path.splitext(filename)
					if extension == '.jpg':
						corp_img = img[j0*crop_size:h,j1*crop_size:w,:]
					elif extension == '.png':
						corp_img = img[j0 * crop_size:h, j1 * crop_size:w]
					else:
						print("error extension:",extension)
					newpath = filepath.replace(basepath, newbasepath)
					newpath = newpath.replace(imagepath, newimagepath)
					if not os.path.exists(newpath):
						os.makedirs(newpath)
					newimagefile = os.path.join(newpath, filename + '_crop_{0}_{1}{2}'.format(j0, j1, extension))
					imageio.imwrite(newimagefile, corp_img)
					newimage_files.append(newimagefile)

	return newimage_files


filter = [".jpg", ".JPG"]  # 设置过滤后的文件类型 当然可以设置多个类型
imagepath = 'image_1024'

def all_path(dirname):
	result = []  # 所有的文件
	for maindir, subdir, file_name_list in os.walk(dirname):
		# print("1:",maindir) #当前主目录
		# print("2:",subdir) #当前主目录下的所有目录
		# print("3:",file_name_list)  #当前主目录下的所有文件
		# print("maindir",maindir.split('/',-1)[-1])
		if maindir.split('/', -1)[-1] == imagepath:
			# print(maindir)
			for filename in file_name_list:
				apath = os.path.join(maindir, filename)  # 合并成一个完整路径
				# result.append(apath)
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


	basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset'
	labelpath = 'label_1024_dilat'
	labelpath_gt = 'groundtruth_1024_erode'
	random_flag = True
	x_train_filenames, y_train_filenames, y1_train_filenames = all_filenames(basepath, labelpath_gt, labelpath,
																			 random_flag)

	newbasepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/train_dataset_512'
	newlabelpath = 'label_512_dilat'
	newlabelpath_gt = 'groundtruth_512_erode'
	newimagepath = 'image_512'
	# crop_jpg_png(x_train_filenames, basepath, newbasepath, imagepath, newimagepath)
	# crop_jpg_png(y_train_filenames, basepath, newbasepath, labelpath_gt, newlabelpath_gt)
	# crop_jpg_png(y1_train_filenames, basepath, newbasepath, labelpath, newlabelpath)

	basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset'
	labelpath = 'label_1024'
	labelpath_gt = 'groundtruth_1024'
	random = False
	x_test_filenames, y_test_filenames, y1_test_filenames = all_filenames(basepath, labelpath_gt, labelpath, random)

	newbasepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512'
	newlabelpath = 'label_512'
	newlabelpath_gt = 'groundtruth_512'
	newimagepath = 'image_512'
	crop_jpg_png(x_test_filenames, basepath, newbasepath, imagepath, newimagepath)
	crop_jpg_png(y_test_filenames, basepath, newbasepath, labelpath_gt, newlabelpath_gt)
	crop_jpg_png(y1_test_filenames, basepath, newbasepath, labelpath, newlabelpath)



	basepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset'
	labelpath = 'label_1024_dilat'
	labelpath_gt = 'groundtruth_1024_erode'
	random = False
	x_valid_filenames, y_valid_filenames, y1_valid_filenames = all_filenames(basepath, labelpath_gt, labelpath, random)

	newbasepath = '/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/valid_dataset_512'
	newlabelpath = 'label_512_dilat'
	newlabelpath_gt = 'groundtruth_512_erode'
	newimagepath = 'image_512'
	crop_jpg_png(x_valid_filenames, basepath, newbasepath, imagepath, newimagepath)
	crop_jpg_png(y_valid_filenames, basepath, newbasepath, labelpath_gt, newlabelpath_gt)
	crop_jpg_png(y1_valid_filenames, basepath, newbasepath, labelpath, newlabelpath)
