import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.FCN import build_fcn
from models.Unet import build_unet


SUPPORTED_MODELS = ["FCN", "Unet"]



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
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn
