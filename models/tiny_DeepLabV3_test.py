# coding=utf-8
import os
import sys


import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


import imageio
import subprocess





def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net

def AtrousSpatialPyramidPoolingModule(inputs, depth=256):

    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
    image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

    atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

    atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)

    atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)

    atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)
    net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

    return net


def conv_bn_relu(name, x, out_filters, kernel_size, training_flag):
    with tf.variable_scope(name):
        out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME', name='conv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        return out


def deconv_bn_relu(name, x, out_filters, kernel_size, strides=(2, 2), training_flag=None):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(x, out_filters, kernel_size, strides=strides, padding='SAME', name='deconv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        return out


def dense_bn_relu_dropout(name, x, num_neurons, dropout_rate, training_flag):
    with tf.variable_scope(name):
        out = tf.layers.dense(x, num_neurons, name='dense')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        out = tf.layers.dropout(out, dropout_rate, training=training_flag, name='dropout')
        return out


'''
def fcn_module(name, input_x, is_training):
    with tf.variable_scope(name):
        conv1 = conv_bn_relu('conv1_block', input_x, 16, (3, 3), is_training)
        conv2 = conv_bn_relu('conv2_block', conv1, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool1'):
            max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv3 = conv_bn_relu('conv3_block', max_pool1, 32, (3, 3), is_training)
        conv4 = conv_bn_relu('conv4_block', conv3, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool2'):
            max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv5 = conv_bn_relu('conv5_block', max_pool2, 64, (3, 3), is_training)
        conv6 = conv_bn_relu('conv6_block', conv5, 64, (3, 3), is_training)

        with tf.variable_scope('max_pool3'):
            max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv7 = conv_bn_relu('conv7_block', max_pool3, 128, (3, 3), is_training)

        de_conv3 = deconv_bn_relu('deconv3_block', conv7, 64, (3, 3), (2, 2), training_flag=is_training)
        conv8 = conv_bn_relu('conv8_block', de_conv3, 64, (3, 3), is_training)

        de_conv2 = deconv_bn_relu('deconv2_block', conv8, 32, (3, 3), (2, 2), training_flag=is_training)
        conv9 = conv_bn_relu('conv9_block', de_conv2, 32, (3, 3), is_training)

        de_conv1 = deconv_bn_relu('deconv1_block', conv9, 16, (3, 3), (2, 2), training_flag=is_training)
        conv10 = conv_bn_relu('conv10_block', de_conv1, 16, (3, 3), is_training)
        out = conv_bn_relu('conv11_block', conv10, 2, (1, 1), is_training)
    return out
'''

def base_module(name, input_x, is_training):
    with tf.variable_scope(name):
        conv1 = conv_bn_relu('conv1_block', input_x, 16, (3, 3), is_training)
        conv2 = conv_bn_relu('conv2_block', conv1, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool1'):
            max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv3 = conv_bn_relu('conv3_block', max_pool1, 32, (3, 3), is_training)
        conv4 = conv_bn_relu('conv4_block', conv3, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool2'):
            max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv5 = conv_bn_relu('conv5_block', max_pool2, 64, (3, 3), is_training)
        conv6 = conv_bn_relu('conv6_block', conv5, 64, (3, 3), is_training)

        with tf.variable_scope('max_pool3'):
            max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv7 = conv_bn_relu('conv7_block', max_pool3, 128, (3, 3), is_training)
        
    return conv7

def build_deeplabv3(inputs, num_classes, is_training=True):
    print("inputs",inputs.shape)
    #logits, end_points, frontend_scope = build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    end_points = base_module(name='tiny_deeplabv3/base_module', input_x = inputs, is_training=is_training)
    
    label_size = tf.shape(inputs)[1:3]
    print("tf.shape(inputs)",tf.shape(inputs)[0])
    print("tf.shape(inputs)",tf.shape(inputs)[1])
    print("tf.shape(inputs)",tf.shape(inputs)[2])
    print("tf.shape(inputs)",tf.shape(inputs)[3])
    print("label_size",label_size)

    with tf.variable_scope('tiny_deeplabv3/left_module'):
        net = AtrousSpatialPyramidPoolingModule(end_points,depth=64)
        print("end_points",end_points.shape)
        print("net1",net.shape)
    
        net = Upsampling(net, label_size)
        print("net2",net.shape)
        #net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
        net = conv_bn_relu('logits', net, num_classes, (1, 1), is_training)
        print("net3",net.shape)
    '''
    de_conv3 = deconv_bn_relu('deconv3_block', net, 64, (3, 3), (2, 2), training_flag=is_training)
    conv8 = conv_bn_relu('conv8_block', de_conv3, 64, (3, 3), is_training)

    de_conv2 = deconv_bn_relu('deconv2_block', conv8, 32, (3, 3), (2, 2), training_flag=is_training)
    conv9 = conv_bn_relu('conv9_block', de_conv2, 32, (3, 3), is_training)

    de_conv1 = deconv_bn_relu('deconv1_block', conv9, 16, (3, 3), (2, 2), training_flag=is_training)
    conv10 = conv_bn_relu('conv10_block', de_conv1, 16, (3, 3), is_training)

    net = conv_bn_relu('logits', conv10, num_classes, (1, 1), is_training)
    '''
    return net






if __name__ == '__main__':
    image_path = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/HpDataSet_20x/test_dataset_1_new/splite_Top8192_left12288_Bottom12288_Right16384_scale40_0.jpg'
    #image = imageio.imread(image_path)
    #image = image.reshape([1] + list(image.shape))
    #image = np.asarray(image,dtype=np.float32)
    x_list=[]
    x_list.append(imageio.imread(image_path))
    x_list.append(imageio.imread(image_path))
    image = np.asarray(x_list,dtype=np.float32)
    print("image",image.shape)
    
    with tf.variable_scope('inputs'):            
        #self_x = tf.placeholder(tf.float32, [None, None, None, 3])
        self_x = tf.placeholder(tf.float32, [2, 512, 512, 3])
        self_y = tf.placeholder(tf.int64, [None, None, None])
        self_y1 = tf.placeholder(tf.int64, [None, None, None])
        self_is_training = tf.placeholder(tf.bool, name='Training_flag')            
    tf.add_to_collection('inputs', self_x)
    tf.add_to_collection('inputs', self_y)
    tf.add_to_collection('inputs', self_y1)
    tf.add_to_collection('inputs', self_is_training)

    #model_name = "DeepLabV3"
    is_training = True
    num_classes = 2
    #self_out = build_deeplabv3(inputs=self_x, num_classes=num_classes, preset_model = model_name, frontend="ResNet101", is_training=is_training)
    self_out = build_deeplabv3(inputs=self_x, num_classes=num_classes, is_training=is_training)

    
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())
    picout = sess.run(self_out, feed_dict={self_x: image})
    print("picout",picout.shape)
