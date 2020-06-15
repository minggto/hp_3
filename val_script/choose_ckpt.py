import os
import sys

import numpy as np
import cv2
import time, datetime
import argparse

from glob import glob
import imageio
import copy

import pickle

# f = open(os.path.join(statistic_path, 'statistics.pckl'), 'wb')
# pickle.dump([TFF, P1, P21, R2, R12], f)
def load_data(statistic_path):
	f = open(os.path.join(statistic_path, 'statistics.pckl'), 'rb')
	TFF, P1, P21, R2, R12 = pickle.load(f)
	return TFF, P1, P21, R2, R12


def ratio(x1):
	r_x1 = (np.array(x1,np.float)-np.min(x1))/(np.max(x1)-np.min(x1))
	return r_x1

def cal(P,R):
	P = np.asarray(P)
	R = np.asarray(R)
	f1 = 2*P*R/(P+R)
	# num = np.argmax(f1)
	f2 = (1+2**2)*P*R/(2**2*P+R)
	# num = np.argmax(f2)
	f3 = (1+3**2)*P*R/(3**2*P+R)
	# num = np.argmax(f3)
	num = np.argmax(R)

	return num,P[num],R[num],f1[num]

def sortmax(nums):
	import heapq
	# nums = [1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]
	# 最大的3个数的索引
	max_num_index_list = map(nums.index, heapq.nlargest(50, nums))
	# 最小的3个数的索引
	# min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))
	# print(list(max_num_index_list))
	return list(max_num_index_list)


def plotpic(yr1, yp1,yr2, yp2, yrm, ypm, model):
	import matplotlib.pyplot as plt


	plt.figure(1)
	plt.subplot(131)
	plt.title(model)
	plt.xlabel("ckpt")
	# plt.ylabel("y")
	# plt.xlim(xmax=2.1, xmin=0)  # 改变xlim——改变x轴的取值范围
	# plt.ylim(ymax=1, ymin=0.4)
	# new_ticks = np.linspace(0, 2.0, 5)
	# plt.xticks(new_ticks)  # 改变xticks——坐标轴显示和精度
	# x = [0.05, 0.2, 0.4, 1, 2]


	x = list(range(441,641))
	f1a = 2.0 * yr1 * yp1 / (yr1 + yp1)
	f1b = 2.0 * yr2 * yp2 / (yr2 + yp2)
	f1c = 2.0 * yrm * ypm / (yrm + ypm)
	# b = 4
	# fb = (1.0 + b) * yr * yp / (yr + b * yp)

	# plt.plot(x, yr, '--^', color='b', label='Recall')
	# plt.plot(x, yp, '--s', color='g', label='Precision')
	# plt.plot(x, f1, '--d', color='y', label='F1')
	# plt.plot(x, fb, '--o', color='r', label='Fβ')

	plt.plot(x, yr1, '--v', color='b', label='single1_recall')
	plt.plot(x, yr2, '--*', color='m', label='single2_recall')
	plt.plot(x, yrm, '--d', color='y', label='multi_recall')
	plt.legend()  # show label

	plt.subplot(132)
	plt.title(model)
	plt.xlabel("ckpt")
	plt.plot(x, yp1, '--v', color='b', label='single1_precision')
	plt.plot(x, yp2, '--*', color='m', label='single2_precision')
	plt.plot(x, ypm, '--d', color='y', label='multi_precision')
	plt.legend()  # show label

	plt.subplot(133)
	plt.title(model)
	plt.xlabel("ckpt")
	plt.plot(x, f1a, '--v', color='b', label='single1_f1')
	plt.plot(x, f1b, '--*', color='m', label='single2_f1')
	plt.plot(x, f1c, '--d', color='y', label='multi_f1')
	# plt.plot(x, TPE_ASPP_f1, '--s', color='k', label='2PE-ASPP F1')
	# plt.plot(x, THPE_f1, '--^', color='g', label='3PE F1')
	# plt.plot(x, THPE_ASPP_f1, '--o', color='r', label='3PE-ASPP F1')

	plt.legend()  # show label

	plt.show()





if __name__ == '__main__':

	statistic_path = ['./statistic_path/FCN_statistic_12m','./statistic_path/Unet_statistic_12m','./statistic_path/tiny_deeplabv3_statistic_12m','./statistic_path/pspnet_statistic_12m','./statistic_path/DANet_statistic_12m']
	for k in range(len(statistic_path)):
		TFF, P1, P21, R2_t, R12 = load_data(statistic_path[k])
		P1 = []
		P21 = []
		Pm1 = []
		R2 = []
		R12 = []
		Rm2 = []
		# print(len(TFF))
		#exit()
		for i in range(200):
			j = len(TFF)-200+i
			# print(j,TFF[j])
			tffl11, tffl21, tfflm1, tffl22, tffl12, tfflm2 = TFF[j]
			#print(tffl11, tffl21, tfflm1, tff22, tffl12, tfflm2)
			#exit()
			P1.append(tffl11[1])
			P21.append(tffl21[1])
			Pm1.append(tfflm1[1])
			# R2.append(tffl22[0])
			R12.append(tffl12[0])
			Rm2.append(tfflm2[0])
			R2.append(R2_t[j])

			pred1 = tffl11[0]+tffl11[1]
			pred2 = tffl21[0] + tffl21[1]
			predm = tfflm1[0] + tfflm1[1]
			label2_pos = tffl12[0]+tffl12[2]

		# rp1 = 1.0-ratio(P1)
		# rp2 = 1.0-ratio(P21)
		# rpm = 1.0-ratio(Pm1)
		# rr1 = ratio(R12)
		# rr2 = ratio(R2)
		# rrm = ratio(Rm2)
		# n1,Pa,Ra,f1a = cal(rp1, rr1)
		# n2,Pb,Rb,f1b = cal(rp2, rr2)
		# nm,Pc,Rc,f1c = cal(rpm, rrm)

		preci1 = 1.0-np.array(P1)/pred1
		preci2 = 1.0-np.array(P21)/pred2
		precim = 1.0 - np.array(Pm1) / predm
		recal1 = np.array(R12)/label2_pos
		recal2 = np.array(R2) / label2_pos
		recalm = np.array(Rm2) / label2_pos
		n1, Pa, Ra, f1a = cal(1.0-np.array(P1)/pred1, np.array(R12)/label2_pos)
		n2, Pb, Rb, f1b = cal(1.0-np.array(P21)/pred2, np.array(R2)/label2_pos)
		nm, Pc, Rc, f1c = cal(1.0-np.array(Pm1)/predm, np.array(Rm2)/label2_pos)

		model = (statistic_path[k].split('/')[-1]).split('_')[0]
		print(model,n1,n2,nm,'            ->',n1+441,n2+441,nm+441)
		# print("Pa,Ra,f1a",Pa,Ra,f1a)
		# print("Pb, Rb, f1b", Pb, Rb, f1b)
		# print("Pc,Rc,f1c", Pc,Rc,f1c)

		# model = (statistic_path[k].split('/')[-1]).split('_')[0]
		# plotpic(yr1 = recal1, yp1 = preci1,yr2 = recal2 , yp2 = preci2, yrm = recalm, ypm = precim, model = model)





	# for k in range(len(statistic_path)):
	# 	TFF, P1, P21, R2_t, R12 = load_data(statistic_path[k])
	# 	P1 = []
	# 	P21 = []
	# 	Pm1 = []
	# 	R2 = []
	# 	R12 = []
	# 	Rm2 = []
	# 	# print(len(TFF))
	# 	#exit()
	# 	for i in range(200):
	# 		j = len(TFF)-200+i
	# 		# print(j,TFF[j])
	# 		tffl11, tffl21, tfflm1, tffl22, tffl12, tfflm2 = TFF[j]
	# 		#print(tffl11, tffl21, tfflm1, tff22, tffl12, tfflm2)
	# 		#exit()
	# 		P1.append(tffl11[1])
	# 		P21.append(tffl21[1])
	# 		Pm1.append(tfflm1[1])
	# 		# R2.append(tffl22[0])
	# 		R12.append(tffl12[0])
	# 		Rm2.append(tfflm2[0])
	# 		R2.append(R2_t[j])
	#
	# 	rp1 = 1.0-ratio(P1)
	# 	rp2 = 1.0-ratio(P21)
	# 	rpm = 1.0-ratio(Pm1)
	# 	rr1 = ratio(R2)
	# 	rr2 = ratio(R12)
	# 	rrm = ratio(Rm2)
	# 	# print(type(rr1.tolist()))
	# 	arg_list_r1 = sortmax(rr1.tolist())
	# 	arg_list_r2 = sortmax(rr2.tolist())
	# 	arg_list_rm = sortmax(rrm.tolist())
	# 	r1 = []
	# 	r2 = []
	# 	rm = []
	# 	p1 = []
	# 	p2 = []
	# 	pm = []
	# 	for i in range(len(arg_list_r1)):
	# 		r1.append(rr1[arg_list_r1[i]])
	# 		p1.append(rp1[arg_list_r1[i]])
	# 	for i in range(len(arg_list_r2)):
	# 		r2.append(rr2[arg_list_r2[i]])
	# 		p2.append(rp2[arg_list_r2[i]])
	# 	for i in range(len(arg_list_rm)):
	# 		rm.append(rrm[arg_list_rm[i]])
	# 		pm.append(rpm[arg_list_rm[i]])
	#
	# 	n1 = cal(p1, r1)
	# 	n2 = cal(p2, r2)
	# 	nm = cal(pm, rm)
	# 	n1,n2,nm = arg_list_r1[n1], arg_list_r2[n2], arg_list_rm[nm]
	# 	print(n1,n2,nm,'            ->',n1+441,n2+441,nm+441)

#f1
# FCN 73 173 185             -> 514 614 626
# Unet 15 159 37             -> 456 600 478
# tiny_deeplabv3 177 193 119      -> 618 634 560
# pspnet 138 25 123          -> 579 466 564
# DANet 127 169 187          -> 568 610 628
# add 441
#f2
# FCN 97 173 101            -> 538 614 542
# Unet 15 127 37            -> 456 568 478
# tiny_deeplabv3 109 173 113      -> 550 614 554
# pspnet 138 5 21        -> 579 446 462
# DANet 127 169 185         -> 568 610 626
#f3
# 161 173 101             -> 602 614 542
# 109 127 173             -> 550 568 614
# 109 173 113             -> 550 614 554
# 138 5 21             -> 579 446 462
# 127 169 135             -> 568 610 576

#recall
# 161 173 167             -> 602 614 608
# 137 127 135             -> 578 568 576
# 161 173 113             -> 602 614 554
# 166 5 53             -> 607 446 494
# 137 169 33             -> 578 610 474

#TOP50-F1
# 161 173 101             -> 602 614 542
# 109 127 173             -> 550 568 614
# 109 173 113             -> 550 614 554
# 138 5 21             -> 579 446 462

#f1
# FCN 127 73 101             -> 568 514 (542)
# Unet 173 21 37             -> 614 462 (478)
# tiny 133 161 113             -> 574 602 (554)
# pspnet 199 95 53             -> (640) 536 494
# DANet 175 131 135             -> 616 572 (576)

##f2
# FCN 173 23 167             -> 614 464 608
# Unet 127 91 135             -> 568 532 576
# tiny 133 59 113             -> (574) 500 (554)
# pspnet 5 47 53             -> 446 488 (494)
# DANet 169 75 33             -> 610 516 474

# ##f3
# FCN 173 23 167             -> (614 464 608)
# Unet 127 16 135             -> (568) 457 (576)
# tiny 173 59 113             -> 614 (500 554)
# pspnet 5 47 53             -> (446 488 494)
# DANet 169 137 33             -> (610) 578 (474)
#
# ##recall
# FCN 173 23 167             -> 614 464 608
# Unet 127 16 135             -> 568 457 576
# tiny 173 59 113             -> 614 500 554
# pspnet 5 166 53             -> 446 607 494
# DANet 169 41 33             -> 610 482 474
