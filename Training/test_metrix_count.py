# @Time     : 2020/6/8 下午3:45
# @Auther   : Yiming Yang   
# @FileName : test_metrix.py
from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import pickle
import random
import os, sys
import subprocess
import collections
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

from utils import metrics

from glob import glob
import imageio

import xlrd
import xlwt
from xlutils.copy import copy
import py_excel.py_excel as excel



def calculation(n, ncl=2):
	t = np.asarray([0 for i in range(ncl)])
	for i in range(ncl):
		for j in range(ncl):
			t[i] = t[i] + n[i, j]

	sum_nii = 0
	for i in range(ncl):
		sum_nii = sum_nii + n[i, i]
	# sum_ti=t[0]+t[1]
	sum_ti = np.sum(t)
	pa = sum_nii / sum_ti

	sum_nt = 0
	fix_ncl = ncl
	for i in range(ncl):
		if t[i] == 0:  # 如果t[i]为0，则第i类的数量为0，即此样本为特殊样本，此样本中不存在第i类标签，那么修正后此样本的标签数量-1
			sum_nt = sum_nt
			fix_ncl = fix_ncl - 1
		else:
			sum_nt = sum_nt + n[i, i] / t[i]
	mpa = (1.0 / fix_ncl) * sum_nt

	IU = np.asarray([0.0 for i in range(ncl)])
	fix_ncl = ncl
	for i in range(ncl):
		sum_nji = 0
		for j in range(ncl):
			sum_nji = sum_nji + n[j, i]
		if t[i] + sum_nji - n[i, i] > 0:
			IU[i] = n[i, i] / (t[i] + sum_nji - n[i, i])
			if t[i] == 0:  # 如果t[i]为0，则第i类的数量为0，即此样本为特殊样本，此样本中不存在第i类标签，那么修正后此样本的标签数量-1
				fix_ncl = ncl - 1
		else:
			IU[i] = 1.0
	# print("i=",i,"IU[i]:",IU[i],"fenmu:",t[i]+sum_nji-n[i,i],"fenzi n[i,i]:",n[i,i])
	# mIU=(1.0/ncl)*np.sum(IU)
	mIU = (1.0 / fix_ncl) * np.sum(IU)

	wIU = np.asarray([0.0 for i in range(ncl)])
	for i in range(ncl):
		sum_nji = 0
		for j in range(ncl):
			sum_nji = sum_nji + n[j, i]
		if t[i] + sum_nji - n[i, i] > 0:
			wIU[i] = t[i] * n[i, i] / (t[i] + sum_nji - n[i, i])
		else:
			wIU[i] = t[i] * 1.0
	fwIU = np.sum(wIU) / np.sum(t)

	# print("pa", pa, "mpa", mpa, "IU", IU, "mIU", mIU, "fwIU", fwIU)
	return pa, mpa, IU, mIU, fwIU


def calculation2(n, ncl=2):
	'''
	t=np.asarray([0 for i in range(ncl)])
	for i in range(ncl):
		for j in range(ncl):
			t[i]=t[i]+n[i,j]
	'''
	tp = n[1, 1]
	fn = n[1, 0]
	fp = n[0, 1]
	tn = n[0, 0]
	acc = np.asarray([0.0 for i in range(ncl)])
	if tp == 0:
		recall = 0
		precision = 0
		f1 = 0
		acc[0] = tn / (tn + fp)
		acc[1] = 0
	else:
		recall = tp / (tp + fn)
		precision = tp / (tp + fp)
		f1 = (2 * precision * recall) / (recall + precision)
		acc[0] = tn / (tn + fp)
		acc[1] = tp / (fn + tp)
	pa = (tp + tn) / (tn + fp + fn + tp)
	mpa = (acc[0] + acc[1]) / 2
	# print("recall", recall, "precision", precision)
	return recall, precision, f1, acc, pa, mpa


def count(label_temp, gpu_temp):
	# H = label_temp.shape[1]
	# W = label_temp.shape[2]
	n = [[0, 0], [0, 0]]
	n = np.asarray(n, dtype='float32')

	# comp_pd = pd.DataFrame(gpu_temp[0, :, :])
	# result = comp_pd.apply(pd.value_counts)
	# array_result = np.array(result.index)
	# result = np.sum(result, axis=1)
	# # print("result",result)
	#
	# comp_pd = pd.DataFrame(label_temp[0, :, :])
	# result = comp_pd.apply(pd.value_counts)
	# array_result = np.array(result.index)
	# result = np.sum(result, axis=1)
	# # print("result",result)

	# comp = label_temp[0, :, :] * 10 + gpu_temp[0, :, :]
	comp = label_temp * 10 + gpu_temp
	comp_pd = pd.DataFrame(comp)
	result = comp_pd.apply(pd.value_counts)
	array_result = np.array(result.index)
	result = np.sum(result, axis=1)
	# print("result",result)

	if 0 in array_result:
		n[0, 0] = result[0]
	else:
		n[0, 0] = 0
	if 1 in array_result:
		n[0, 1] = result[1]
	else:
		n[0, 1] = 0
	if 10 in array_result:
		n[1, 0] = result[10]
	else:
		n[1, 0] = 0
	if 11 in array_result:
		n[1, 1] = result[11]
	else:
		n[1, 1] = 0

	# print("n",n)
	return n



def inference(inference_path, visualpath):
	# print("come in inference", inference_path)
	inference_files = glob(os.path.join(inference_path, '*.png'))
	list_sample_num = []
	n = [[0, 0], [0, 0]]
	n = np.asarray(n, dtype='float32')
	for i in range(len(inference_files)):
		file_path = inference_files[i]
		fname, fename = os.path.split(file_path)
		# prename, postfix = os.path.splitext(fename)
		pred_path = os.path.join(visualpath,fename)

		label_image = imageio.imread(file_path)
		pred_image = imageio.imread(pred_path)

		sample_num = count(label_temp=label_image//255, gpu_temp=pred_image//255)
		list_sample_num.append(sample_num)
		n = n + sample_num

	return list_sample_num,n



def write_excel(book_name_xls,list_row):
	# book_name_xls = 'data0.xls'
	sheet_name_xls = 'data00'
	value_title = [['model_task','ckpt','pos_iou','mean_iou','fwIU','recall','precision','f1','pa','mpa'], ]
	value1 = list_row
	if not os.path.exists(book_name_xls):
		excel.write_excel_xls(book_name_xls, sheet_name_xls, value_title)
	excel.write_excel_xls_append(book_name_xls, value1)




parser = argparse.ArgumentParser()
parser.add_argument('--model_task', type=str, default="FCN_multi", help='model_task name.')
parser.add_argument('--statistic_path', type=str, default="", help='statistic_path name.')
parser.add_argument('--checkpoint_i', type=int, default=639, help='checkpoint_i')
args = parser.parse_args()


'''
'/2_data/share/workspace/yym/HP/hp3_visural/pspnet_single1/split4096_vec_8_takegt/0579/takegt/'
'/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/split4096_vec_8/groundtruth_512/'
'''
if __name__ == '__main__':
	labelpath = 'groundtruth_512'
	test_basepath = "/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/"
	dirs = os.listdir(test_basepath)

	model_task = args.model_task
	visual_basepath = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task
	checkpoint_i = args.checkpoint_i
	num = '%04d' % checkpoint_i

	sumnn = [[0, 0], [0, 0]]
	sumn = np.asarray(sumnn, dtype='float32')
	for j in range(len(dirs)):
		basepath = test_basepath + dirs[j] + '/' + labelpath
		visualpath = visual_basepath +  '/' + dirs[j] + '_takegt' + '/' + num + '/takegt/'
		list_sample_num,n = inference(basepath, visualpath)

		pa, mpa, IU, mIU, fwIU = calculation(n, ncl=2)
		recall, precision, f1, acc, pa, mpa = calculation2(n, ncl=2)
		print("{}:IU, mIU, fwIU, recall, precision, f1, acc, pa, mpa".format(dirs[j]),IU, mIU, fwIU, recall, precision, f1, acc, pa, mpa)

		sumn = sumn+n
	pa, mpa, IU, mIU, fwIU = calculation(sumn, ncl=2)
	recall, precision, f1, acc, pa, mpa = calculation2(sumn, ncl=2)
	print(model_task,num,"IU, mIU, fwIU, recall, precision, f1, acc, pa, mpa", IU, mIU, fwIU, recall, precision, f1, acc, pa, mpa)

	# write_excel(book_name_xls= '/2_data/share/workspace/yym/HP/hp3_visural/test_metrix_count.xls',list_row = [[model_task, num, IU[1], mIU, fwIU, recall, precision, f1, pa, mpa],])
	write_excel(book_name_xls= '/2_data/share/workspace/yym/HP/hp3_visural/test_metrix_count.xls',list_row = [[model_task, num, str(IU[1]), str(mIU), str(fwIU), str(recall),str(precision), str(f1), str(pa), str(mpa)],])


	# writefile = '/2_data/share/workspace/yym/HP/hp3_visural/test_metrix_count.txt'
	# with open(writefile, 'a+') as f:
	# 	f.write(model_task)
	# 	f.write(' ')
	# 	f.write("IU, mIU, fwIU, recall, precision, f1, acc, pa, mpa")
	# 	f.write(' ')
	# 	f.write(IU)
	# 	f.write(' ')
	# 	f.write(mIU)
	# 	f.write(' ')
	# 	f.write(fwIU)
	# 	f.write(' ')
	# 	f.write(recall)
	# 	f.write(' ')
	# 	f.write(precision)
	# 	f.write(' ')
	# 	f.write(f1)
	# 	f.write(' ')
	# 	f.write(acc)
	# 	f.write(' ')
	# 	f.write(pa)
	# 	f.write(' ')
	# 	f.write(mpa)
	# 	f.write('\n')



	pkl_path = '/2_data/share/workspace/yym/HP/hp3_visural/'
	file_name = pkl_path + model_task + '_' + num + '_test_metrix_count.pkl'
	with open(file_name, 'wb')as f:
		pickle.dump([pa, mpa, IU, mIU, fwIU, recall, precision, f1, acc, sumn], f)
		# pickle.dump({'IU': x_test_filenames,
		# 			 'y_test': y_test_filenames,
		# 			 'z_test': y1_test_filenames,
		# 			 }, f)
