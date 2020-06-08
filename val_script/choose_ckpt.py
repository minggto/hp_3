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
	f1 = P*R/(P+R)
	num = np.argmax(f1)
	return num

if __name__ == '__main__':
	statistic_path = './statistic_path/pspnet_statistic_12m'
	TFF, P1, P21, R2, R12 = load_data(statistic_path)
	P1 = []
	P21 = []
	Pm1 = []
	R2 = []
	R12 = []
	Rm2 = []
	print(len(TFF))
	#exit()
	for i in range(200):
		j = len(TFF)-200+i
		print(j,TFF[j])
		tffl11, tffl21, tfflm1, tffl22, tffl12, tfflm2 = TFF[j]
		#print(tffl11, tffl21, tfflm1, tff22, tffl12, tfflm2)
		#exit()
		P1.append(tffl11[1])
		P21.append(tffl21[1])
		Pm1.append(tfflm1[1])
		R2.append(tffl22[0])
		R12.append(tffl12[0])
		Rm2.append(tfflm2[0])

	rp1 = 1.0-ratio(P1)
	rp2 = 1.0-ratio(P21)
	rpm = 1.0-ratio(Pm1)
	rr1 = ratio(R2)
	rr2 = ratio(R12)
	rrm = ratio(Rm2)
	n1 = cal(rp1, rr1)
	n2 = cal(rp2, rr2)
	nm = cal(rpm, rrm)
	print(n1,n2,nm)
# fcn 73 173 185             -> 514 614 626
# Unet 15 159 37             -> 456 600 478
# deeplabv3 177 193 119      -> 618 634 560
# pspnet 138 25 123          -> 579 466 564
# add 441


