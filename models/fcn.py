import tensorflow as tf
import imageio
import numpy as np


def fcn_module(name, net_input, num_classes, is_training):
    #with tf.variable_scope('network'):
    with tf.variable_scope(name):
        conv1 = conv_bn_relu('conv1_block', net_input, 16, (3, 3), is_training)
        conv2 = conv_bn_relu('conv2_block', conv1, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool1'):
            max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')
            # print('max_pool output shape: {}'.format(max_pool1.shape))

        conv3 = conv_bn_relu('conv3_block', max_pool1, 32, (3, 3), is_training)
        conv4 = conv_bn_relu('conv4_block', conv3, 32, (3, 3), is_training)

        with tf.variable_scope('max_pool2'):
            max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv5 = conv_bn_relu('conv5_block', max_pool2, 64, (3, 3), is_training)
        conv6 = conv_bn_relu('conv6_block', conv5, 64, (3, 3), is_training)

        with tf.variable_scope('max_pool3'):
            max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(2, 2), name='max_pool')

        conv7 = conv_bn_relu('conv7_block', max_pool3, 128, (3, 3), is_training)

        de_conv3 = deconv_bn_relu('deconv3_block', conv7, 64, (3, 3), (2, 2), is_training)
        conv8 = conv_bn_relu('conv8_block', de_conv3, 64, (3, 3), is_training)

        de_conv2 = deconv_bn_relu('deconv2_block', conv8, 32, (3, 3), (2, 2), is_training)
        conv9 = conv_bn_relu('conv9_block', de_conv2, 32, (3, 3), is_training)

        de_conv1 = deconv_bn_relu('deconv1_block', conv9, 16, (3, 3), (2, 2), is_training)
        conv10 = conv_bn_relu('conv10_block', de_conv1, 16, (3, 3), is_training)
        out = conv_bn_relu('conv11_block', conv10, num_classes, (1, 1), is_training)
        return out


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



if __name__ == '__main__':
    '''
    vec = 'split4096_vec_1'
    image_file = 'splite_Top8192_left8192_Bottom12288_Right12288_scale40_14_180'
    image_path = '/1_data/yym_workspace/UnicNet/data/hp/hp_data/thesis_data/train_dataset_augmentation/{}/image_512/'.format(vec)
    img_image = imageio.imread(image_path+image_file+'.jpg')
    
    vec_img = []
    vec_img.append(img_image)
    vec_img.append(img_image)
    input_img = np.asarray(vec_img)
    print("input_img.shape",input_img.shape)
    '''

    #self_x = tf.placeholder(tf.float32, [1, None, None, 3])
    self_x = tf.placeholder(tf.float32, [1, 512, 512, 3])
    model = fcn_module(name='fcn',net_input=self_x, num_classes=2, is_training=False)
    print("model",model)
    #print("model",model.conv1)
    #print("model",model.conv2)
    #print("model",model.max_pool1)
    '''
    init = tf.global_variables_initializer()
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    sess.run(init)
    '''
    
    #print("model",model)
    






