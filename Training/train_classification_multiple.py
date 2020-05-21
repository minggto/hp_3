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
parser.add_argument('--crop_height', type=int, default=1024, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=47, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=0.1, help='Whether to randomly change the image brightness')
parser.add_argument('--rotation', type=float, default=False, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FCN", help='The model you are using. See model_builder.py for supported models')

parser.add_argument('--statistic_path', type=str, default="./", help='The model you are using. See model_builder.py for supported models')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def data_augmentation(input_image, output_image1, output_image2):
    # Data augmentation

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image1 = cv2.flip(output_image1, 1)
        output_image2 = cv2.flip(output_image2, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image1 = cv2.flip(output_image1, 0)
        output_image2 = cv2.flip(output_image2, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image1 = cv2.warpAffine(output_image1, M, (output_image1.shape[1], output_image1.shape[0]), flags=cv2.INTER_NEAREST)
        output_image2 = cv2.warpAffine(output_image2, M, (output_image2.shape[1], output_image2.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image1, output_image2


num_classes = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output1 = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
net_output2 = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network1, init_fn = model_builder.build_model(model_name=args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network1, labels=net_output1))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network1, labels=net_output2))
loss = 0.5*loss1+0.5*loss2

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
opt = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss, var_list=[var for var in tf.trainable_variables()])
#opt = tf.train.RMSPropOptimizer(learning_rate=0.00005, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
opt = tf.group([opt, update_ops])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
ckpt_path = args.ckpt
model_checkpoint_name = ckpt_path+"/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)
    #saver1.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, train_target_names, val_input_names, val_output_names, val_target_names = utils.prepare_data_multiple(args.dataset_pkl_train, args.dataset_pkl_val)

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
print("")

avg_loss_per_epoch = []

avg_loss_per_epoch1 = []
avg_prec_per_epoch1 = []
avg_rec_per_epoch1 = []
avg_f1_per_epoch1 = []
pos_accuracy_per_epoch1 = []
mean_iou_per_epoch1 = []
global_accuracy_per_epoch1 = []

avg_loss_per_epoch2 = []
avg_prec_per_epoch2 = []
avg_rec_per_epoch2 = []
avg_f1_per_epoch2 = []
pos_accuracy_per_epoch2 = []
mean_iou_per_epoch2 = []
global_accuracy_per_epoch2 = []

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
        output_image_batch1 = []
        output_image_batch2 = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image1 = utils.load_image(train_output_names[id])
            output_image2 = utils.load_image(train_target_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image1, output_image2 = data_augmentation(input_image, output_image1, output_image2)


                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image1 = np.float32(helpers.one_hot_it(label=output_image1, label_values=[0,255]))
                output_image2 = np.float32(helpers.one_hot_it(label=output_image2, label_values=[0,255]))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch1.append(np.expand_dims(output_image1, axis=0))
                output_image_batch2.append(np.expand_dims(output_image2, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch1 = output_image_batch1[0]
            output_image_batch2 = output_image_batch2[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch1 = np.squeeze(np.stack(output_image_batch1, axis=1))
            output_image_batch2 = np.squeeze(np.stack(output_image_batch2, axis=1))

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output1:output_image_batch1,net_output2:output_image_batch2})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    #avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%(ckpt_path,epoch)):
        os.makedirs("%s/%04d"%(ckpt_path,epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(ckpt_path,epoch))


    if epoch % args.validation_step == 0:
        avg_loss_per_epoch.append(mean_loss)

        print("Performing validation")
        target1=open("%s/%04d/val_scores.csv1"%(ckpt_path,epoch),'w')
        target1.write("val_name, precision, recall, f1 score, pos_accuracy, mean_iou, global_accuracy, %s\n" % ('hp, background'))
        target2=open("%s/%04d/val_scores.csv2"%(ckpt_path,epoch),'w')
        target2.write("val_name, precision, recall, f1 score, pos_accuracy, mean_iou, global_accuracy, %s\n" % ('hp, background'))


        prec_list1 = []
        rec_list1 = []
        f1_list1 = []
        pos_accuracy_list1 = []
        mean_iou_list1 = []
        global_accuracy_list1 = []

        prec_list2 = []
        rec_list2 = []
        f1_list2 = []
        pos_accuracy_list2 = []
        mean_iou_list2 = []
        global_accuracy_list2 = []


        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            gt1 = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt1 = helpers.reverse_one_hot(helpers.one_hot_it(gt1, [0,255]))
            gt2 = utils.load_image(val_target_names[ind])[:args.crop_height, :args.crop_width]
            gt2 = helpers.reverse_one_hot(helpers.one_hot_it(gt2, [0,255]))

            # st = time.time()

            output_image1 = sess.run(network1,feed_dict={net_input:input_image})
            #output_image2 = sess.run(network2,feed_dict={net_input:input_image})


            output_image1 = np.array(output_image1[0,:,:,:])
            output_image1 = helpers.reverse_one_hot(output_image1)
            out_vis_image1 = helpers.colour_code_segmentation(output_image1, [0,255])

            prec1, rec1, f11, pos_accuracy1, mean_iou1, global_accuracy1 = utils.evaluate_segmentation_new(pred=output_image1, label=gt1, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target1.write("%s, %f, %f, %f, %f, %f, %f"%(file_name, prec1, rec1, f11, pos_accuracy1, mean_iou1, global_accuracy1))

            if math.isnan(prec1):
                prec1 = 0
            prec_list1.append(prec1)
            if math.isnan(rec1):
                rec1 = 0
            rec_list1.append(rec1)
            if math.isnan(f11):
                f11 = 0
            f1_list1.append(f11)
            if math.isnan(pos_accuracy1):
                pos_accuracy1 = 0
            pos_accuracy_list1.append(pos_accuracy1)
            if math.isnan(mean_iou1):
                mean_iou1 = 0
            mean_iou_list1.append(mean_iou1)
            if math.isnan(global_accuracy1):
                global_accuracy1 = 0
            global_accuracy_list1.append(global_accuracy1)

            gt1 = helpers.colour_code_segmentation(gt1, [0,255])


            prec2, rec2, f12, pos_accuracy2, mean_iou2, global_accuracy2 = utils.evaluate_segmentation_new(pred=output_image1, label=gt2, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target2.write("%s, %f, %f, %f, %f, %f, %f"%(file_name, prec2, rec2, f12, pos_accuracy2, mean_iou2, global_accuracy2))


            if math.isnan(prec2):
                prec2 = 0
            prec_list2.append(prec2)
            if math.isnan(rec2):
                rec2 = 0
            rec_list2.append(rec2)
            if math.isnan(f12):
                f12 = 0
            f1_list2.append(f12)
            if math.isnan(pos_accuracy2):
                pos_accuracy2 = 0
            pos_accuracy_list2.append(pos_accuracy2)
            if math.isnan(mean_iou2):
                mean_iou2 = 0
            mean_iou_list2.append(mean_iou2)
            if math.isnan(global_accuracy2):
                global_accuracy2 = 0
            global_accuracy_list2.append(global_accuracy2)

            gt2 = helpers.colour_code_segmentation(gt2, [0,255])


            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_pred1.png"%(ckpt_path,epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image1), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt1.png"%(ckpt_path,epoch, file_name),cv2.cvtColor(np.uint8(gt1), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt2.png"%(ckpt_path,epoch, file_name),cv2.cvtColor(np.uint8(gt2), cv2.COLOR_RGB2BGR))


        target1.close()
        target2.close()

        avg_prec_per_epoch1.append(np.mean(prec_list1))
        avg_rec_per_epoch1.append(np.mean(rec_list1))
        avg_f1_per_epoch1.append(np.mean(f1_list1))
        pos_accuracy_per_epoch1.append(np.mean(pos_accuracy_list1))
        mean_iou_per_epoch1.append(np.mean(mean_iou_list1))
        global_accuracy_per_epoch1.append(np.mean(global_accuracy_list1))

        print("\n Average validation accuracy for epoch # %04d = %f"% (epoch, np.mean(global_accuracy_list1)))
        print("Validation IoU score = ", np.mean(mean_iou_list1))
        print("Validation precision = ", np.mean(prec_list1))
        print("Validation recall = ", np.mean(rec_list1))
        print("Validation F1 score = ", np.mean(f1_list1))
        print("Validation accuracy = ", np.mean(pos_accuracy_list1))

        avg_prec_per_epoch2.append(np.mean(prec_list2))
        avg_rec_per_epoch2.append(np.mean(rec_list2))
        avg_f1_per_epoch2.append(np.mean(f1_list2))
        pos_accuracy_per_epoch2.append(np.mean(pos_accuracy_list2))
        mean_iou_per_epoch2.append(np.mean(mean_iou_list2))
        global_accuracy_per_epoch2.append(np.mean(global_accuracy_list2))

        print("\n Average validation accuracy for epoch # %04d = %f"% (epoch, np.mean(global_accuracy_list2)))
        print("Validation IoU score = ", np.mean(mean_iou_list2))
        print("Validation precision = ", np.mean(prec_list2))
        print("Validation recall = ", np.mean(rec_list2))
        print("Validation F1 score = ", np.mean(f1_list2))
        print("Validation accuracy = ", np.mean(pos_accuracy_list2))


    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)


    l = len(global_accuracy_per_epoch1)
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(l), global_accuracy_per_epoch1)
    ax1.set_title("Average validation global accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")
    plt.savefig(os.path.join(args.statistic_path,'global_accuracy1_vs_epochs.png'))

    plt.clf()

    fig11, ax11 = plt.subplots(figsize=(11, 8))
    ax11.plot(range(l), mean_iou_per_epoch1)
    ax11.set_title("Average validation IoU vs epochs")
    ax11.set_xlabel("Epoch")
    ax11.set_ylabel("Avg. val. IoU")
    plt.savefig(os.path.join(args.statistic_path,'IoU1_vs_epochs.png'))

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(l), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(os.path.join(args.statistic_path,'loss_vs_epochs.png'))

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(range(l), avg_prec_per_epoch1)
    ax3.set_title("Precision vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Precision")
    plt.savefig(os.path.join(args.statistic_path,'precision1_vs_epochs.png'))

    plt.clf()

    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.plot(range(l), avg_rec_per_epoch1)
    ax4.set_title("Recall vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Current Recall")
    plt.savefig(os.path.join(args.statistic_path,'recall1_vs_epochs.png'))

    plt.clf()

    fig5, ax5 = plt.subplots(figsize=(11, 8))
    ax5.plot(range(l), avg_f1_per_epoch1)
    ax5.set_title("F1 vs epochs")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Current f1")
    plt.savefig(os.path.join(args.statistic_path,'f11_vs_epochs.png'))

    plt.clf()

    fig6, ax6 = plt.subplots(figsize=(11, 8))
    ax6.plot(range(l), pos_accuracy_per_epoch1)
    ax6.set_title("Pos accuracy vs epochs")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Current pos accuracy")
    plt.savefig(os.path.join(args.statistic_path,'pos_accuracy1_vs_epochs.png'))

    
    l = len(global_accuracy_per_epoch2)
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(l), global_accuracy_per_epoch2)
    ax1.set_title("Average validation global accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")
    plt.savefig(os.path.join(args.statistic_path,'global_accuracy2_vs_epochs.png'))

    plt.clf()

    fig11, ax11 = plt.subplots(figsize=(11, 8))
    ax11.plot(range(l), mean_iou_per_epoch2)
    ax11.set_title("Average validation IoU vs epochs")
    ax11.set_xlabel("Epoch")
    ax11.set_ylabel("Avg. val. IoU")
    plt.savefig(os.path.join(args.statistic_path,'IoU2_vs_epochs.png'))

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(range(l), avg_prec_per_epoch2)
    ax3.set_title("Precision vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Precision")
    plt.savefig(os.path.join(args.statistic_path,'precision2_vs_epochs.png'))

    plt.clf()

    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.plot(range(l), avg_rec_per_epoch2)
    ax4.set_title("Recall vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Current Recall")
    plt.savefig(os.path.join(args.statistic_path,'recall2_vs_epochs.png'))

    plt.clf()

    fig5, ax5 = plt.subplots(figsize=(11, 8))
    ax5.plot(range(l), avg_f1_per_epoch2)
    ax5.set_title("F1 vs epochs")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Current f1")
    plt.savefig(os.path.join(args.statistic_path,'f12_vs_epochs.png'))

    plt.clf()

    fig6, ax6 = plt.subplots(figsize=(11, 8))
    ax6.plot(range(l), pos_accuracy_per_epoch2)
    ax6.set_title("Pos accuracy vs epochs")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Current pos accuracy")
    plt.savefig(os.path.join(args.statistic_path,'pos_accuracy2_vs_epochs.png'))


    import pickle
    f1=open(os.path.join(args.statistic_path,'statistics1.pckl'), 'wb')
    pickle.dump([avg_loss_per_epoch1, avg_prec_per_epoch1, avg_rec_per_epoch1, avg_f1_per_epoch1, pos_accuracy_per_epoch1, mean_iou_per_epoch1, global_accuracy_per_epoch1], f1)
    f1.close()
    f2=open(os.path.join(args.statistic_path,'statistics2.pckl'), 'wb')
    pickle.dump([avg_loss_per_epoch2, avg_prec_per_epoch2, avg_rec_per_epoch2, avg_f1_per_epoch2, pos_accuracy_per_epoch2, mean_iou_per_epoch2, global_accuracy_per_epoch2], f2)
    f2.close()






