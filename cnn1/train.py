import tensorflow as tf
import cv2
import numpy as np
import dataset
import time
from datetime import timedelta
import math
import random

# path to traing data
train_path = '/home/kcsmta/Desktop/WORKS/SPECT/code/data/data_training'
MODEL_NAME = 'spect-image-model'

# class info
classes = ['0', '1', '2', '3', '4']
num_classes = len(classes)

# size input image
img_size = 224
witdh = img_size
height = img_size
channels = 3

# Convolution params
num_conv1_filters = 96
conv1_kernel_size = 11
conv1_stride = 4

num_conv2_filters = 256
conv2_kernel_size = 3
conv2_stride = 2

num_conv3_filters = 384
conv3_kernel_size = 3
conv3_stride = 1

num_conv4_filters = 384
conv4_kernel_size = 3
conv4_stride = 1

num_conv5_filters = 256
conv5_kernel_size = 3
conv5_stride = 1

num_fully1_nodes = 4096
num_fully2_nodes = 4096

# validation split
validation_size = .2

# read training data
data = dataset.read_train_sets(train_path, img_size, classes, validation_size)

def create_weights(shape):
	initializer = tf.contrib.layers.xavier_initializer()
	return tf.Variable(initializer(shape))
def create_bias(size):
	return tf.Variable(tf.constant(0.05,shape=size))
def convolution_layer(input,num_input_channels,num_filters,kernel_size,kernel_stride,use_pool):
	weights = create_weights(shape=[kernel_size,kernel_size,num_input_channels,num_filters])
	bias = create_bias(num_filters)
	layer = tf.nn.conv2d(input,weights,strides=[1,kernel_stride,kernel_stride,1],padding="SAME")
	layer += bias
	layer = tf.nn.relu(layer)
	if(use_pool == True):
		layer = tf.nn.max_pool(layer,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
	return layer


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer[1:4].num_elements()
	layer = tf.reshape(layer,[-1,num_features])
def fc_layer(input,num_outputs):
	return tf.contrib.layers.fully_connected(input,num_output)

X = tf.placeholder(tf.float32,shape=[None,witdh,height,channels],name = 'x')
y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
# y_true_cls = tf.argmax(y_true,dimesion=1)
y_true_cls = tf.argmax(y_true)

conv1 = convolution_layer(input = X , num_input_channels= channels,num_filters=num_conv1_filters,kernel_size=conv1_kernel_size,kernel_stride=conv1_stride,use_pool=True)
# conv2 = convolution_layer(input = conv1,num_input_channels=num_conv1_filters,num_filters=num_conv2_filters ,kernel_size=conv2_kernel_size,kernel_stride=conv2_stride,use_pool=True)
# conv3 = convolution_layer(conv2,num_conv2_filters,num_conv3_filters,conv3_kernel_size,conv3_stride,False)
# conv4 = convolution_layer(conv3,num_conv3_filters,num_conv4_filters,conv4_kernel_size,conv4_stride,False)
# conv5 = convolution_layer(conv4,num_conv4_filters,num_conv5_filters,conv5_kernel_size,conv5_stride,False)

# conv5_fc = flatten_layer(conv5)

# fc1 = tf.contrib.layers.fully_connected(conv5_fc,num_fully1_nodes)
# fc2 = tf.contrib.layers.fully_connected(fc1,num_fully2_nodes)

# output = tf.contrib.layer.fully_connected(fc2,num_classes)

# y_pred = tf.nn.softmax(output,name='y_pred')
# y_pred_cls = tf.argmax(y_pred,dimesion=1)

# init = tf.global_variables_initializer()
# x_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output,lables=y_true)
# cost = tf.reduce_mean(x_entropy)
# optimizer = tf.train.AdamOptimizer().minimize(cost)
# correct_prediction = tf.equal(y_pred,y_true)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# saver = tf.train.Saver()
# batch_size = 50
# with tf.Session as sess:
# 	sess.run(init)
# 	saver.restore('')
# 	for i in range(3000):
# 		x_batch,y_true_batch, _ ,cls_batch = dataset.train.next_batch(batch_size)
# 		x_valid_batch, y_valid_batch,_,valid_cls_batch = data.valid.next_batch(batch_size)
# 		feed_dict_tr = {x:x_batch,y_true:y_true_batch}
# 		feed_dict_val = {x:x_valid_batch,y_true:y_true_batch}

# 		sess.run(optimizer,feed_dict=feed_dict_tr)
# 		if i % int(data.train.num_examples/batch_size) == 0:
# 			val_loss = sess.run(cost,feed_dict=feed_dict_val)
# 			epoch = int(i/ int(data.train.num_examples/batch_size))
# 			acc = sess.run(accuracy,feed_dict=feed_dict_tr)
# 			val_acc = sess.run(accuracy,feed_dict=feed_dict_val)
# 			msg = "Training epoch {0} ---- Training acc {1:>6.1%} ---- Validation acc {2:>6.1%} ---- Validation loss {3:3f}"
# 			print(msg.format(epoch+1,acc,val_acc,val_loss))
# 			saver.Saver(sess,MODEL_NAME)
