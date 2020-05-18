import tensorflow as tf
from tensorflow.contrib import slim
import imageio
import numpy as np
# from models.other_model import resnet_v2
#from other_model import resnet_v2

def build_frontend(inputs, frontend, is_training=True):
    if frontend == 'ResNet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            frontend_scope='resnet_v2_50'
    elif frontend == 'ResNet101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            frontend_scope='resnet_v2_101'
    elif frontend == 'ResNet152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            frontend_scope='resnet_v2_152'
    else:
        raise ValueError("Unsupported fronetnd model '%s'. This function only supports ResNet50, ResNet101, ResNet152" % (frontend))

    return logits, end_points, frontend_scope


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


"""
Network Architecture
"""


def MultiFcnModel(net_input, num_classes, is_training):
    feature_map_size = tf.shape(net_input)


    with tf.variable_scope('MultiFcn_3'):
        conv1 = conv_bn_relu('conv1_block', net_input, 16, (3, 3), is_training)
        conv2 = conv_bn_relu('conv2_block', conv1, 16, (3, 3), is_training)
        max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool1')


        conv3 = conv_bn_relu('conv3_block', max_pool1, 32, (3, 3), is_training)
        conv4 = conv_bn_relu('conv4_block', conv3, 32, (3, 3), is_training)
        max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool2')

        conv5 = conv_bn_relu('conv5_block', max_pool2, 64, (3, 3), is_training)
        conv6 = conv_bn_relu('conv6_block', conv5, 64, (3, 3), is_training)
        max_pool3 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(2, 2), name='max_pool3')

        conv7 = conv_bn_relu('conv7_block', max_pool3, 128, (3, 3), is_training)
        conv8 = conv_bn_relu('conv8_block', conv7, 128, (3, 3), is_training)

        de_conv3 = deconv_bn_relu('deconv3_block', conv8, 64, (3, 3), (2, 2), training_flag=is_training)

        conv9 = conv_bn_relu('conv9_block', de_conv3, 64, (3, 3), is_training)
        conv10 = conv_bn_relu('conv10_block', conv9, 64, (3, 3), is_training)

        de_conv2 = deconv_bn_relu('deconv2_block', conv10, 32, (3, 3), (2, 2), training_flag=is_training)

        conv11 = conv_bn_relu('conv11_block', de_conv2, 32, (3, 3), is_training)
        conv12 = conv_bn_relu('conv12_block', conv11, 32, (3, 3), is_training)

        de_conv1 = deconv_bn_relu('deconv1_block', conv12, 16, (3, 3), (2, 2), training_flag=is_training)

        conv13 = conv_bn_relu('conv13_block', de_conv1, 16, (3, 3), is_training)
        conv14 = conv_bn_relu('conv14_block', conv13, 16, (3, 3), is_training)
        out3 = tf.image.resize_bilinear(conv14, (feature_map_size[1], feature_map_size[2]))


    with tf.variable_scope('out'):
        conv1 = conv_bn_relu('conv1_block', out3, 16, (3, 3), is_training)
        out = conv_bn_relu('conv3_block', conv1, num_classes, (1, 1), is_training)


    '''
    print('conv10   output shape: ',conv10.shape)
    print('conv13   output shape: ',conv13.shape)
    print('conv14   output shape: ',conv14.shape)
    print('out2      output shape: ',out2.shape)
    print('out3      output shape: ',out3.shape)
    print('out4      output shape: ',out4.shape)
    print('concat_1 output shape: ',concat_1.shape)
    print('out      output shape: ',out.shape)
    '''

    return out


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

def psp_module(name, input_x, is_training):
    pool_factors = [1, 2, 3, 6]
    channel = 128
    feature_map_shape = tf.shape(input_x)[1:3]
    with tf.variable_scope(name):
        pool_list = []
        up_list = []
        for i in range(4):
            with tf.variable_scope('pooling_{}'.format(i)):
                pool_temp = tf.layers.average_pooling2d(input_x, pool_size=(input_x.shape[1]//pool_factors[i], input_x.shape[2]//pool_factors[i]), strides=(input_x.shape[1]//pool_factors[i], input_x.shape[2]//pool_factors[i]))
                print("---pool_temp.shape---", pool_temp.shape)
                conv_temp = conv_bn_relu('conv_block_{}'.format(i), pool_temp, channel, (1, 1), is_training)
                print("---conv_temp.shape---", conv_temp.shape)
                pool_list.append(conv_temp)
        with tf.variable_scope('Upsampling'):
            for i in range(4):
                with tf.variable_scope('BI_{}'.format(i)):
                    up_temp = tf.image.resize_images(pool_list[i], size=feature_map_shape, method=0)
                    up_list.append(up_temp)

            concat_all = tf.concat([up_list[0], up_list[1], up_list[2],up_list[3], input_x],3,name='concat_all')
            print("---concat_all.shape---", concat_all.shape)
            return concat_all


def pspnet(inputs, num_classes, is_training=True):
    print("inputs", inputs.shape)

    end_points = base_module(name='pspnet/base_module', input_x=inputs, is_training=is_training)

    label_size = tf.shape(inputs)[1:3]
    print("tf.shape(inputs)", tf.shape(inputs)[0])
    print("tf.shape(inputs)", tf.shape(inputs)[1])
    print("tf.shape(inputs)", tf.shape(inputs)[2])
    print("tf.shape(inputs)", tf.shape(inputs)[3])
    print("label_size", label_size)

    convx = psp_module(name='pspnet/psp_module', input_x = end_points, is_training=is_training)

    with tf.variable_scope('out'):
        out_feature_map_shape = tf.shape(inputs)[1:3]
        convx = conv_bn_relu('conv1_block', convx, 128, (1, 1), is_training)
        convx = tf.image.resize_bilinear(convx, size=out_feature_map_shape)
        convx = conv_bn_relu('conv2_block', convx, num_classes, (1, 1), is_training)
    return convx





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
    # model = MultiFcnModel(net_input=self_x, num_classes=2, is_training=False)
    model = pspnet(inputs=self_x, num_classes=2, is_training=False)
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
    

