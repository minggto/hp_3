from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import collections
import math

import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="-1", help='GPU you are using.')
parser.add_argument('--ckpt', type=str, default="", help='ckpt save path.')
parser.add_argument('--save_first_ckpt', type=str2bool, default=False, help='Save the first ckpt flag.')
parser.add_argument('--first_ckpt', type=str, default="", help='The path of first ckpt.')

parser.add_argument('--num_epochs', type=int, default=640, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')

parser.add_argument('--continue_training', type=str2bool, default=True, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="HP", help='Dataset you are using.')
parser.add_argument('--dataset_pkl_train', type=str, default=" ", help='Training data pkl.')
parser.add_argument('--dataset_pkl_val', type=str, default=" ", help='Val data pkl.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=47, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=0.1, help='Whether to randomly change the image brightness')
parser.add_argument('--rotation', type=float, default=False, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FCN", help='The model you are using. See model_builder.py for supported models')

parser.add_argument('--statistic_path', type=str, default="./", help='The model you are using. See model_builder.py for supported models')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def data_augmentation(input_image, output_image):
    # Data augmentation

    #input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

num_classes = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])
#opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
opt = tf.group([opt, update_ops])

ckpt_path = args.ckpt
model_checkpoint_name = ckpt_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt"

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
if args.save_first_ckpt:
    import gc
    print("Saving the first checkpoint")
    first_ckpt_path = args.first_ckpt
    saver.save(sess, first_ckpt_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt")
    gc.collect()
    exit()

utils.count_params()


if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired

if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names = utils.prepare_data_single2(args.dataset_pkl_train, args.dataset_pkl_val)

print(train_input_names[0])
print(train_output_names[0])
print(val_input_names[0])
print(val_output_names[0])

print(len(train_output_names))
print(len(val_output_names))

print(train_input_names[-1])
print(train_output_names[-1])
print(val_input_names[-1])
print(val_output_names[-1])


print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tRotation -->", args.rotation)
print("\tbrightness -->",args.brightness)
print("")

avg_loss_per_epoch = []
avg_prec_per_epoch = []
avg_rec_per_epoch = []
avg_f1_per_epoch = []
pos_accuracy_per_epoch = []
mean_iou_per_epoch = []
global_accuracy_per_epoch = []


# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)


                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=[0,255]))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%(ckpt_path,epoch)):
        os.makedirs("%s/%04d"%(ckpt_path,epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(ckpt_path,epoch))


    if epoch % args.validation_step == 0:

        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%(ckpt_path,epoch),'w')
        target.write("val_name, precision, recall, f1 score, pos_accuracy, mean_iou, global_accuracy, %s\n" % ('hp,background'))


        prec_list = []
        rec_list = []
        f1_list = []
        pos_accuracy_list = []
        mean_iou_list = []
        global_accuracy_list = []


        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = helpers.one_hot_it(gt, [0,255])
            gt = helpers.reverse_one_hot(gt)

            # st = time.time()

            output_image = sess.run(network,feed_dict={net_input:input_image})


            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, [0,255])

            prec, rec, f1, pos_accuracy, mean_iou, global_accuracy = utils.evaluate_segmentation_new(pred=output_image, label=gt, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f, %f"%(file_name, prec, rec, f1, pos_accuracy, mean_iou, global_accuracy))


            if math.isnan(prec):
                prec = 0
            prec_list.append(prec)
            if math.isnan(rec):
                rec = 0
            rec_list.append(rec)
            if math.isnan(f1):
                f1 = 0
            f1_list.append(f1)
            if math.isnan(pos_accuracy):
                pos_accuracy = 0
            pos_accuracy_list.append(pos_accuracy)
            if math.isnan(mean_iou):
                mean_iou = 0
            mean_iou_list.append(mean_iou)
            if math.isnan(global_accuracy):
                global_accuracy = 0
            global_accuracy_list.append(global_accuracy)

            gt = helpers.colour_code_segmentation(gt, [0,255])

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred.png"%(ckpt_path,epoch, file_name), np.uint8(out_vis_image))
            cv2.imwrite("%s/%04d/%s_gt.png"%(ckpt_path,epoch, file_name), np.uint8(gt))


        target.close()

        avg_loss_per_epoch.append(mean_loss)
        avg_prec_per_epoch.append(np.mean(prec_list))
        avg_rec_per_epoch.append(np.mean(rec_list))
        avg_f1_per_epoch.append(np.mean(f1_list))
        pos_accuracy_per_epoch.append(np.mean(pos_accuracy_list))
        mean_iou_per_epoch.append(np.mean(mean_iou_list))
        global_accuracy_per_epoch.append(np.mean(global_accuracy_list))

        print("\n Average validation accuracy for epoch # %04d = %f"% (epoch, np.mean(global_accuracy_list)))
        print("Validation IoU score = ", np.mean(mean_iou_list))
        print("Validation precision = ", np.mean(prec_list))
        print("Validation recall = ", np.mean(rec_list))
        print("Validation F1 score = ", np.mean(f1_list))
        print("Validation accuracy = ", np.mean(pos_accuracy_list))


    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []

    if not os.path.exists(args.statistic_path):
        os.makedirs(args.statistic_path)

    l = len(global_accuracy_per_epoch)
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(l), global_accuracy_per_epoch)
    ax1.set_title("Average validation global accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")
    plt.savefig(os.path.join(args.statistic_path, 'global_accuracy_vs_epochs.png'))

    plt.clf()

    fig11, ax11 = plt.subplots(figsize=(11, 8))
    ax11.plot(range(l), mean_iou_per_epoch)
    ax11.set_title("Average validation IoU vs epochs")
    ax11.set_xlabel("Epoch")
    ax11.set_ylabel("Avg. val. IoU")
    plt.savefig(os.path.join(args.statistic_path,'IoU_vs_epochs.png'))

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(l), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(os.path.join(args.statistic_path,'loss_vs_epochs.png'))

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(range(l), avg_prec_per_epoch)
    ax3.set_title("Precision vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Precision")
    plt.savefig(os.path.join(args.statistic_path,'precision_vs_epochs.png'))

    plt.clf()

    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.plot(range(l), avg_rec_per_epoch)
    ax4.set_title("Recall vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Current Recall")
    plt.savefig(os.path.join(args.statistic_path,'recall_vs_epochs.png'))

    plt.clf()

    fig5, ax5 = plt.subplots(figsize=(11, 8))
    ax5.plot(range(l), avg_f1_per_epoch)
    ax5.set_title("F1 vs epochs")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Current f1")
    plt.savefig(os.path.join(args.statistic_path,'f1_vs_epochs.png'))

    plt.clf()

    fig6, ax6 = plt.subplots(figsize=(11, 8))
    ax6.plot(range(l), pos_accuracy_per_epoch)
    ax6.set_title("Pos accuracy vs epochs")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Current pos accuracy")
    plt.savefig(os.path.join(args.statistic_path,'pos_accuracy_vs_epochs.png'))


    import pickle
    f=open(os.path.join(args.statistic_path,'statistics.pckl'), 'wb')
    pickle.dump([avg_loss_per_epoch, avg_prec_per_epoch, avg_rec_per_epoch, avg_f1_per_epoch, pos_accuracy_per_epoch, mean_iou_per_epoch, global_accuracy_per_epoch], f)
    f.close()










