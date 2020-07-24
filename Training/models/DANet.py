import tensorflow as tf
from tensorflow.contrib import slim
import imageio
import numpy as np


def conv_bn_relu(name, x, out_filters, kernel_size, training_flag):
    with tf.variable_scope(name):
        out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        # print('{} output shape: {}'.format(name, out.shape))
        return out


def deconv_bn_relu(name, x, out_filters, kernel_size, strides=(2, 2), training_flag=None):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(x, out_filters, kernel_size, strides=strides, padding='SAME',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        # print('{} output shape: {}'.format(name, out.shape))
        return out


def dense_bn_relu_dropout(name, x, num_neurons, dropout_rate, training_flag):
    with tf.variable_scope(name):
        out = tf.layers.dense(x, num_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense')
        out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
        out = tf.nn.relu(out)
        out = tf.layers.dropout(out, dropout_rate, training=training_flag, name='dropout')
        return out



def PAM_Module(x, is_training):
    with tf.variable_scope('PAM'):
        k1 = tf.Variable(tf.constant(0.1),name='k1',dtype = tf.float32,trainable=False)
        #tf.nn.conv2d(x,)
        x = tf.cast(x,dtype = tf.float32)
        m_batchsize, height, width, C = x.shape
        m_batchsize = -1
        height = 64
        width = 64
        conv1 = conv_bn_relu('conv1_block', x, C//8, (1, 1), is_training)
        conv2 = conv_bn_relu('conv2_block', x, C//8, (1, 1), is_training)
        conv3 = conv_bn_relu('conv3_block', x, C, (1, 1), is_training)

        x = tf.transpose(x,perm=[0, 3, 1, 2])

        proj_query = tf.transpose(tf.reshape(conv1,[m_batchsize, C//8, width*height]),perm=[0, 2, 1])
        proj_key = tf.reshape(conv2,[m_batchsize, C//8, width*height])
        energy = tf.matmul(proj_query,proj_key)
        # print(energy.shape)
        attention = tf.nn.softmax(energy)

        proj_value = tf.reshape(conv3,[m_batchsize, C, width*height])
        out = tf.matmul(proj_value,tf.transpose(attention,perm=[0, 2, 1]))
        out = tf.reshape(out,[m_batchsize, C, height, width])
        out = k1*out + x
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        return k1,out

def CAM_Module(x):
    with tf.variable_scope('CAM'):
        k2 = tf.Variable(tf.constant(0.1), name='k2', dtype=tf.float32, trainable=False)  # trainable：默认为True，可以后期被算法优化的。如果不想该变量被优化，改为False。
        x = tf.cast(x, dtype=tf.float32)
        x = tf.transpose(x,perm=[0, 3, 1, 2])
        m_batchsize, C, height, width = x.shape
        m_batchsize = -1
        height = 64
        width = 64
        # print(x.shape)
        print("m_batchsize, C, height, width",m_batchsize, C, height, width)

        proj_query = tf.reshape(x, [m_batchsize, C, width*height])
        proj_key = tf.transpose(tf.reshape(x, [m_batchsize, C, width*height]), perm=[0, 2, 1])
        energy = tf.matmul(proj_query, proj_key)
        # print(energy.shape)
        energy_new = tf.reduce_max(energy, [-1], keep_dims=True)
        # print(energy_new.shape)
        energy_new = tf.tile(energy_new, [1, 1, energy.shape[2]])
        energy_new = energy_new - energy
        # print(energy_new.shape)

        attention = tf.nn.softmax(energy_new)
        proj_value = tf.reshape(x, [m_batchsize, C, width*height])
        out = tf.matmul(attention, proj_value)
        print("attention",attention.shape)
        print("proj_value",proj_value.shape)
        print("out",out.shape)
        out = tf.reshape(out, [m_batchsize, C, height, width])

        # print(k2.dtype)
        # print((k2 * out).dtype)
        out = tf.multiply(k2, out) + x
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        return k2,out

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


def DA_module(name, input_x, is_training):
    with tf.variable_scope(name):
        convx = conv_bn_relu('conv1_block', input_x, 128, (3, 3), is_training)
        k1, pam_out = PAM_Module(convx, is_training)
        k2, cam_out = CAM_Module(convx)
        print("*********************convx, pam_out, cam_out: shape***********************************")
        print(pam_out.shape,cam_out.shape)
        convx = tf.concat([pam_out, cam_out], 3, name='concat_1')
    return convx


def build_DANet(inputs, preset_model, num_classes, is_training=True):
    if preset_model != "DANet":
        raise ValueError("Unsupported model '%s'. This function only supports tiny_deeplabv3" % (preset_model))
    print("inputs", inputs.shape)

    end_points = base_module(name='DANet/base_module', input_x=inputs, is_training=is_training)
    print("end_points", end_points.shape)

    convx = DA_module(name='DANet/DA_module', input_x = end_points, is_training=is_training)

    with tf.variable_scope('out'):
        out_feature_map_shape = tf.shape(inputs)[1:3]
        convx = conv_bn_relu('conv1_block', convx, 128, (3, 3), is_training)
        convx = tf.image.resize_bilinear(convx, size=out_feature_map_shape)
        #convx = conv_bn_relu('conv2_block', convx, num_classes, (1, 1), is_training)
        out = slim.conv2d(convx, num_classes, [1, 1], activation_fn=None, scope='logits')
    return out


if __name__ == '__main__':

    self_x = tf.placeholder(tf.float32, [None, None, None, 3])
    #self_x = tf.placeholder(tf.float32, [8, 512, 512, 3])
    model = build_DANet(inputs=self_x, preset_model="DANet", num_classes=2, is_training=False)
    print("model",model)
    #print("model",model.conv1)

