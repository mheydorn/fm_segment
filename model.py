import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import re
import glob
import scipy.misc
import os
import pickle
from IPython import embed
import cv2
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import datetime
import time
import os
import random
import glob

NUM_OF_CLASSES = 2

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class FCN_Model:
    def __init__(self, image, mask):
        #conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 3, 8])
            b_conv1 = bias_variable([8])
            h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)


        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 8, 16])
            b_conv2 = bias_variable([16])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # conv3
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 16, 32])
            b_conv3 = bias_variable([32])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)


        # conv4
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([3, 3, 32, 64])
            b_conv4 = bias_variable([64])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

        # conv5
        with tf.variable_scope('conv5'):
            W_conv5 = weight_variable([3, 3, 64, 128])
            b_conv5 = bias_variable([128])
            h_conv5 = (conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)

        #Upscale
        with tf.variable_scope('tconv4'):
            W_t1 = utils.weight_variable([4, 4 ,h_pool4.get_shape()[3].value,  h_pool5.get_shape()[3].value], name="W_t1")
            b_t1 = utils.bias_variable([h_pool4.get_shape()[3].value], name="b_t1")
            conv_t1 = utils.conv2d_transpose_strided(h_pool5, W_t1, b_t1, output_shape=tf.shape(h_pool4)) #conv_t1 should have shape of h_pool4
            fuse_1 = tf.add(conv_t1, h_pool4, name="fuse_1") #fuse_1 matches shape matches of h_pool4
        with tf.variable_scope('tconv3'):
            W_t2 = utils.weight_variable([4, 4, h_pool3.get_shape()[3].value, h_pool4.get_shape()[3].value], name="W_t2")
            b_t2 = utils.bias_variable([h_pool3.get_shape()[3].value], name="b_t2")
            conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(h_pool3))
            fuse_2 = tf.add(conv_t2, h_pool3, name="fuse_2") # fuse_2 should match h_pool3 shape
        with tf.variable_scope('tconv2'):
            W_t3 = utils.weight_variable([4, 4, h_pool2.get_shape()[3].value, h_pool3.get_shape()[3].value], name="W_t3")
            b_t3 = utils.bias_variable([h_pool2.get_shape()[3].value], name="b_t3")
            conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=tf.shape(h_pool2))
            fuse_3 = tf.add(conv_t3, h_pool2, name="fuse_3") # fuse_3 should match h_pool2 shape

        with tf.variable_scope('tconv1'):
            W_t4 = utils.weight_variable([4, 4, h_pool1.get_shape()[3].value, h_pool2.get_shape()[3].value], name="W_t4")
            b_t4 = utils.bias_variable([h_pool1.get_shape()[3].value], name="b_t4")
            conv_t4 = utils.conv2d_transpose_strided(fuse_3, W_t4, b_t4, output_shape=tf.shape(h_pool1))
            fuse_4 = tf.add(conv_t4, h_pool1, name="fuse_4") # fuse_4 should match h_pool1 shape


        with tf.variable_scope('tconv0'):
            output_shape = tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], NUM_OF_CLASSES])
            W_t5 = utils.weight_variable([4, 4,  NUM_OF_CLASSES, fuse_4.get_shape()[3].value], name="W_t5")
            b_t5 = utils.bias_variable([NUM_OF_CLASSES], name="b_t5")
            conv_t5 = utils.conv2d_transpose_strided(fuse_4, W_t5, b_t5, output_shape = output_shape)

        annotation_pred = tf.argmax(conv_t5, dimension=3, name="prediction")
        self.predictions = tf.expand_dims(annotation_pred, dim=3)
        self.last_conv_layer = conv_t5
        self.softmax =  tf.nn.softmax(conv_t5)
        self.original_image = image
        self.mask = mask
        
        self.loss  = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = conv_t5 , labels =tf.squeeze(mask , squeeze_dims=[3]), name="entropy")))

        #self.accuracy =  tf.metrics.accuracy(
        #    self.mask,
        #    self.predictions,
        #)

        self.accuracy = tf.metrics.mean_iou(
            self.mask,
            self.predictions,
            num_classes = 2,
            weights=None,
            metrics_collections=None,
            updates_collections=None,
            name=None
        )
















