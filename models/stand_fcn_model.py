import tensorflow as tf
import imageio
import numpy as np

'''
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu
            

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")

        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

'''
def fcn_module(name, net_input, num_classes, is_training):
	# with tf.variable_scope('network'):
	with tf.variable_scope('vgg16'):
		conv1 = conv_relu('conv1_1', net_input, 64, (3, 3), is_training)
		conv2 = conv_relu('conv1_2', conv1, 64, (3, 3), is_training)
		with tf.variable_scope('max_pool1'):
			max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='max_pool')
		print('max_pool1 output shape: {}'.format(max_pool1.shape))

		conv3 = conv_relu('conv2_1', max_pool1, 128, (3, 3), is_training)
		conv4 = conv_relu('conv2_2', conv3, 128, (3, 3), is_training)
		with tf.variable_scope('max_pool2'):
			max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), name='max_pool')
		print('max_pool2 output shape: {}'.format(max_pool2.shape))

		conv5 = conv_relu('conv3_1', max_pool2, 256, (3, 3), is_training)
		conv6 = conv_relu('conv3_2', conv5, 256, (3, 3), is_training)
		conv7 = conv_relu('conv3_3', conv6, 256, (3, 3), is_training)
		with tf.variable_scope('max_pool3'):
			max_pool3 = tf.layers.max_pooling2d(conv7, pool_size=(2, 2), strides=(2, 2), name='max_pool')
		print('max_pool3 output shape: {}'.format(max_pool3.shape))

		conv8 = conv_relu('conv4_1', max_pool3, 512, (3, 3), is_training)
		conv9 = conv_relu('conv4_2', conv8, 512, (3, 3), is_training)
		conv10 = conv_relu('conv4_3', conv9, 512, (3, 3), is_training)
		with tf.variable_scope('max_pool4'):
			max_pool4 = tf.layers.max_pooling2d(conv10, pool_size=(2, 2), strides=(2, 2), name='max_pool')
		print('max_pool4 output shape: {}'.format(max_pool4.shape))

		conv11 = conv_relu('conv5_1', max_pool4, 512, (3, 3), is_training)
		conv12 = conv_relu('conv5_2', conv11, 512, (3, 3), is_training)
		conv13 = conv_relu('conv5_3', conv12, 512, (3, 3), is_training)
		with tf.variable_scope('max_pool5'):
			max_pool5 = tf.layers.max_pooling2d(conv13, pool_size=(2, 2), strides=(2, 2), name='max_pool')
		print('max_pool5 output shape: {}'.format(max_pool5.shape))

		fc6 = conv_relu('fc6', max_pool5, 4096, (7, 7), is_training)
		if  is_training:
			fc6 = tf.layers.dropout(fc6, 0.5, training=is_training, name='dropout6')
		print('fc6 output shape: {}'.format(fc6.shape))

		fc7 = conv_relu('fc7', fc6, 4096, (1, 1), is_training)
		if is_training:
			fc7 = tf.layers.dropout(fc7, 0.5, training=is_training, name='dropout7')
		print('fc7 output shape: {}'.format(fc7.shape))

		fc8 = conv_none('fc8', fc7, num_classes, (1, 1), is_training)
		print('fc8 output shape: {}'.format(fc8.shape))
		pred = tf.argmax(fc8, dimension=3, name='pred')
		print('pred output shape: {}'.format(pred.shape))

	if name == 'fcn32x-vgg16':
		with tf.variable_scope(name):
			conv_t1 = deconv_none('conv_t1', fc8, num_classes, (64, 64), (32, 32), is_training)
			print('conv_t1 output shape: {}'.format(conv_t1.shape))
			out = conv_t1

	if name == 'fcn16x-vgg16':
		with tf.variable_scope(name):
			conv_t1 = deconv_none('conv_t1', fc8, num_classes, (4, 4), (2, 2), is_training)
			print('conv_t1 output shape: {}'.format(conv_t1.shape))
			re_channal_pool4 = conv_none('re_channal_pool4', max_pool4, num_classes, (1, 1), is_training)
			fuse_1 = tf.add(conv_t1, re_channal_pool4, name="fuse_1")

			conv_t2 = deconv_none('conv_t2', fuse_1, num_classes, (32, 32), (16, 16), is_training)
			print('conv_t2 output shape: {}'.format(conv_t2.shape))
			out = conv_t2

	if name == 'fcn8x-vgg16':
		with tf.variable_scope(name):
			conv_t1 = deconv_none('conv_t1', fc8, num_classes, (4, 4), (2, 2), is_training)
			print('conv_t1 output shape: {}'.format(conv_t1.shape))
			re_channal_pool4 = conv_none('re_channal_pool4', max_pool4, num_classes, (1, 1), is_training)
			fuse_1 = tf.add(conv_t1, re_channal_pool4, name="fuse_1")

			conv_t2 = deconv_none('conv_t2', fuse_1, num_classes, (4, 4), (2, 2), is_training)
			print('conv_t2 output shape: {}'.format(conv_t2.shape))
			re_channal_pool3 = conv_none('re_channal_pool3', max_pool3, num_classes, (1, 1), is_training)
			fuse_2 = tf.add(conv_t2, re_channal_pool3, name="fuse_2")

			conv_t3 = deconv_none('conv_t3', fuse_2, num_classes, (16, 16), (8, 8), is_training)
			print('conv_t3 output shape: {}'.format(conv_t3.shape))
			out = conv_t3
	return out


def conv_none(name, x, out_filters, kernel_size, training_flag):
	with tf.variable_scope(name):
		out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME', name='conv')
		return out

def conv_relu(name, x, out_filters, kernel_size, training_flag):
	with tf.variable_scope(name):
		out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME', name='conv')
		out = tf.nn.relu(out)
		return out

def conv_bn_relu(name, x, out_filters, kernel_size, training_flag):
	with tf.variable_scope(name):
		out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME', name='conv')
		out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
		out = tf.nn.relu(out)
		return out


def deconv_none(name, x, out_filters, kernel_size, strides=(2, 2), training_flag=None):
	with tf.variable_scope(name):
		out = tf.layers.conv2d_transpose(x, out_filters, kernel_size, strides=strides, padding='SAME', name='deconv')
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

	# self_x = tf.placeholder(tf.float32, [1, None, None, 3])
	self_x = tf.placeholder(tf.float32, [1, 512, 512, 3])
	model = fcn_module(name='fcn8x-vgg16', net_input=self_x, num_classes=2, is_training=False)
	print("model", model)
	# print("model",model.conv1)
	# print("model",model.conv2)
	# print("model",model.max_pool1)
	'''
	init = tf.global_variables_initializer()
	session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_config.gpu_options.allow_growth = True
	sess = tf.Session(config=session_config)
	sess.run(init)
	'''

# print("model",model)







