import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.FCN import build_fcn
from models.Unet import build_unet
from models.tiny_DeepLabV3 import build_deeplabv3
from models.pspnet import build_pspnet
from models.DANet import build_DANet

SUPPORTED_MODELS = ["FCN", "Unet", "tiny_deeplabv3", "pspnet", "DANet"]



def build_model(model_name, net_input, num_classes, crop_width, crop_height, is_training=True):

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	network = None
	init_fn = None
	if model_name == "FCN":
		network = build_fcn(net_input, preset_model=model_name, num_classes=num_classes)
	elif model_name == "Unet":
		network = build_unet(net_input, preset_model=model_name, num_classes=num_classes)
	elif model_name == "tiny_deeplabv3":
		network = build_deeplabv3(net_input, preset_model=model_name, num_classes=num_classes)
	elif model_name == "pspnet":
		network = build_pspnet(net_input, preset_model=model_name, num_classes=num_classes)
	elif model_name == "DANet":
		network = build_DANet(net_input, preset_model=model_name, num_classes=num_classes)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	print("Finish the model ...")
	return network, init_fn
