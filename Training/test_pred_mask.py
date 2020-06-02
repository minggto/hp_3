from __future__ import print_function
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import time, datetime
import argparse
import random

import subprocess
import collections
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from utils import utils, helpers
from builders import model_builder
from glob import glob
import imageio
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="-1", help='GPU you are using.')
parser.add_argument('--ckpt', type=str, default="", help='ckpt save path.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="FCN", help='The model you are using. See model_builder.py for supported models')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu



class HpInference:
	def __init__(self, sess):
		self.sess = sess
		self.ori_image = None
		self.image_shape = None


	def _pre_processing(self, image_path):
		image = imageio.imread(image_path)
		self.ori_image = copy.copy(image)
		# print(self.ori_image.shape)
		self.image_shape = list(image.shape)
		return image.reshape([1] + list(image.shape))

	def _gpu_processing(self, image, network, net_input):
		# output = self.sess.run('out_argmax/out_argmax:0', feed_dict={'IteratorGetNext:0': image,
		#                                                             'inputs/Training_flag:0': False})
		# output_image = self.sess.run('logits/BiasAdd:0', feed_dict={'Placeholder:0': image})

		output_image = self.sess.run(network, feed_dict={net_input: image})

		output_image = np.array(output_image[0, :, :, :])
		# print("output_image",output_image.shape)
		output_image = np.argmax(output_image, axis=-1)
		# print("sum:output_image",np.sum(output_image))
		# print("output_image",output_image.shape)
		output_image = np.asarray(output_image, dtype=np.uint8)
		# path='/home/yym/workspace/yangyiming/UnicNet/thesis_mains/yongquan/pic/'
		# imageio.imwrite(os.path.join(path, '1.jpg'), output_image*255, 'jpeg')
		# output = output_image.reshape([1] + list(output_image.shape))
		return output_image


	def predict(self, image_path, visual_path, checkpoint_i, network, net_input):
		pre_temp = self._pre_processing(image_path)  # self.ori_image 在imagel赋值后的图像是3维（4096，4096，3）
		gpu_temp = self._gpu_processing(pre_temp, network, net_input)
		#print("output_image", gpu_temp.shape)
		self._visualization(gpu_temp, image_path, visual_path, checkpoint_i)


	def _visualization(self, post_temp, image_path, visual_path, checkpoint_i):

		# 写直接预测二值化图
		image_name = os.path.basename(image_path)
		label = np.asarray(post_temp, dtype=np.uint8)
		label_img = copy.copy(label)
		# label_img[:, :, 0][label == 1] = 0
		label_img[:, :][label == 1] = 255
		# label_img[:, :, 2][label == 1] = 0
		label_img[:, :][label == 0] = 0
		binary_label = label_img

		num = '%04d' % checkpoint_i
		if not os.path.exists(visual_path):
			os.makedirs(visual_path)
		if not os.path.exists(visual_path + '/' + num):
			os.makedirs(visual_path + '/' + num)
		imageio.imwrite(os.path.join(visual_path + '/' + num, image_name), binary_label)






def find(path):
	result = []  # 所有的文件
	# os.path.splitext()将文件名和扩展名分开
	# os.path.split（）返回文件的路径和文件名
	fname, fename = os.path.split(path)
	prename, postfix = os.path.splitext(fename)
	# print("fname",fname)
	fpath = fname.split(imagepath, -1)[0]
	# print("split",fpath)
	ffile = fpath + labelpath + '/' + prename + '.png'
	# print(ffile)
	if os.path.exists(ffile):
		# print("true")
		return ffile, fpath
	else:
		print("false")
		return -1


def load_model():
	# Load model
	# tf.reset_default_graph()
	# restore_graph = tf.Graph()
	num_classes = 2
	session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_config.gpu_options.allow_growth = True
	sess = tf.Session(config=session_config)

	net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
	# net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])
	network, init_fn = model_builder.build_model(model_name=args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

	return sess, network, net_input

def restore_model(sess,checkpoint_i):
	# Restore_saver
	# with tf.Session(graph=restore_graph, config=session_config) as sess:
	num = '%04d' % checkpoint_i

	restore_saver=tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	# restore_saver = tf.train.import_meta_graph(args.ckpt + '/' + num + '/model.ckpt.meta')
	restore_saver.restore(sess, tf.train.latest_checkpoint(args.ckpt + '/' + num))


def inference(inference_path, visualpath, checkpoint_i, sess, network, net_input):
	print("come in inference",inference_path)
	inference_files = glob(os.path.join(inference_path, '*.jpg'))
	jpgfile = inference_files
	

	hp_inference = HpInference(sess)

	for i in range(len(jpgfile)):
		file_path = jpgfile[i]
		#print("jpgfile[i]",i,jpgfile[i])
		#label_path = pngfile[i]
		hp_inference.predict(file_path, visualpath, checkpoint_i, network, net_input)









imagepath = 'image_512'
# labelpath='label_1024'
# labelpath = 'groundtruth_1024'


if __name__ == '__main__':
	test_basepath = "/2_data/share/workspace/yym/HP/hp_thesis_3_canny/dataset/test_dataset_512/"

	model_task = args.ckpt.split('/')[-1]
	print("model_task", model_task)
	visual_basepath = '/2_data/share/workspace/yym/HP/hp3_visural/' + model_task

	dirs = os.listdir(test_basepath)

	sess, network, net_input = load_model()
	print("network",network.shape)


	for checkpoint_i in range(1, 641, 2):
		restore_model(sess, checkpoint_i)
		print("finish restore_saver ", checkpoint_i)

		for j in range(len(dirs)):
			basepath = test_basepath + dirs[j] + '/' + imagepath
			visualpath = visual_basepath +  '/' + dirs[j]
			inference(basepath, visualpath, checkpoint_i, sess, network, net_input)



