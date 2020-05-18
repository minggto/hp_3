import tensorflow as tf



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
def UNetModel(net_input, num_classes, is_training):
    conv1 = conv_bn_relu('conv1_block', net_input, 16, (3, 3), is_training)
    conv2 = conv_bn_relu('conv2_block', conv1, 16, (3, 3), is_training)

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
    conv8 = conv_bn_relu('conv8_block', conv7, 128, (3, 3), is_training)

    with tf.variable_scope('max_pool4'):
        max_pool4 = tf.layers.max_pooling2d(conv8, pool_size=(2, 2), strides=(2, 2), name='max_pool')

    conv9 = conv_bn_relu('conv9_block', max_pool4, 256, (3, 3), is_training)
    conv10 = conv_bn_relu('conv10_block', conv9, 256, (3, 3), is_training)

    de_conv4 = deconv_bn_relu('deconv4_block', conv10, 128, (3, 3), (2, 2), training_flag=is_training)
    concat_4 = tf.concat([conv8, de_conv4], 3, name='concat_4')

    conv11 = conv_bn_relu('conv11_block', concat_4, 128, (3, 3), is_training)
    conv12 = conv_bn_relu('conv12_block', conv11, 128, (3, 3), is_training)

    de_conv3 = deconv_bn_relu('deconv3_block', conv12, 64, (3, 3), (2, 2), training_flag=is_training)
    concat_3 = tf.concat([conv6, de_conv3], 3, name='concat_3')

    conv13 = conv_bn_relu('conv13_block', concat_3, 64, (3, 3), is_training)
    conv14 = conv_bn_relu('conv14_block', conv13, 64, (3, 3), is_training)

    de_conv2 = deconv_bn_relu('deconv2_block', conv14, 32, (3, 3), (2, 2), training_flag=is_training)                                                
    concat_2 = tf.concat([conv4, de_conv2], 3, name='concat_2')

    conv15 = conv_bn_relu('conv15_block', concat_2, 32, (3, 3), is_training)
    conv16 = conv_bn_relu('conv16_block', conv15, 32, (3, 3), is_training)

    de_conv1 = deconv_bn_relu('deconv1_block', conv16, 16, (3, 3), (2, 2), training_flag=is_training)                                                
    concat_1 = tf.concat([conv2, de_conv1], 3, name='concat_1')

    conv17 = conv_bn_relu('conv17_block', concat_1, 16, (3, 3), is_training)
    conv18 = conv_bn_relu('conv18_block', conv17, 16, (3, 3), is_training)

    out = conv_bn_relu('conv19_block', conv18, num_classes, (1, 1), is_training)

    return out

