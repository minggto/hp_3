import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv_bn_relu(name, x, out_filters, kernel_size, training_flag=False):
    with tf.variable_scope(name):
        out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        return out

def deconv_bn_relu(name, x, out_filters, kernel_size, strides=(2, 2), training_flag=False):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(x, out_filters, kernel_size, strides=strides, padding='SAME',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        return out

def dense_bn_relu_dropout(name, x, num_neurons, dropout_rate, training_flag=False):
    with tf.variable_scope(name):
        out = tf.layers.dense(x, num_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        out = tf.layers.dropout(out, dropout_rate, training=training_flag, name='dropout')
        return out

def upconv_2D(name, x, out_filters, kernel_size, strides=(2, 2), training_flag=False):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(x, out_filters, kernel_size, strides=strides, padding='SAME',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        return out


def upconv_add(name, inputA, inputB):
    return tf.add(inputA, inputB, name)


def build_unet(inputs, preset_model, num_classes, is_training=True):
    if preset_model != "Unet":
        raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))

    with tf.variable_scope('network'):
        conv1 = conv_bn_relu('conv1_block', inputs, 16, (3, 3), is_training)
        conv2 = conv_bn_relu('conv2_block', conv1, 16, (3, 3), is_training)

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


        de_conv3 = upconv_2D('deconv3_block', conv7, 64, (3, 3), (2, 2), training_flag=is_training)
        short_add3 = upconv_add('shortadd3_block', de_conv3, conv6)
        conv8 = conv_bn_relu('conv8_block', short_add3, 64, (3, 3), is_training)
        
        de_conv2 = upconv_2D('deconv2_block', conv8, 32, (3, 3), (2, 2), training_flag=is_training)
        short_add2 = upconv_add('shortadd2_block', de_conv2, conv4)
        conv9 = conv_bn_relu('conv9_block', short_add2, 32, (3, 3), is_training)
        
        de_conv1 = upconv_2D('deconv1_block', conv9, 16, (3, 3), (2, 2), training_flag=is_training)
        short_add1 = upconv_add('shortadd1_block', de_conv1, conv2)
        conv10 = conv_bn_relu('conv10_block', short_add1, 16, (3, 3), is_training)
        
        out = slim.conv2d(conv10, num_classes, [1, 1], activation_fn=None, scope='logits')

    return out





