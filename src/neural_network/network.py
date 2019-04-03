#!/usr/bin/env python3
# coding: utf8

import sys
import os
import time

import tensorflow as tf
import numpy as np


class ConfigObject:
    def __init__(self, 
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            dropout=0.25,
            test_amount=0.2):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        if test_amount < 0 or test_amount >= 1:
            raise ValueError("Invalid test amount: {test_amount}, must be (0, 1)")
        self.test_amount = test_amount        



def conv2d(inp, shape: tuple, name: str, strides=(1,1,1,1), padding='SAME'):
    """
    Implements a 2d convolution
    """
    with tf.device('/cpu:0'):
        name += "/filters"
        std_dev = np.sqrt(2.0/(shape[0]*shape[1]*shape[3]))
        filters = tf.get_variable(name, shape, 
                initializer=tf.truncated_normal_initializer(stddev=std_dev))
        biases = tf.get_variable(name + '/biases', [shape[-1]],
                initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(tf.nn.conv2d(inp, filters, strides=strides, 
        padding=padding), biases)


def conv3d(inp, shape, name, strides=(1,1,1,1,1), padding='SAME'):
    """
    :param inp: Tensor with shape: [batch, in_depth, in_height, in_width, in_channels]
    :param shape: Shape to make the weight matrix
    :param strides: tuple with 5 elements showing the stride in every dimension
    :param padding: the type of padding to use in the convolution
    """
    with tf.device('/cpu:0'):
        name += "/filters"
        std_dev = np.sqrt(2.0/(shape[0]*shape[1]*shape[2]*shape[4]))
        filters = tf.get_variable(name, shape, initializer=
                tf.truncated_normal_initializer(stddev=std_dev))
        biases = tf.get_variable(name + '/biases', [shape[-1]], initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(tf.nn.conv3d(inp, filters, strides=strides, padding=padding), biases)


def leakyRelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def leakyRelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def CNN_classifier(inp, config, data_location):
        # tf Graph input

    if not isinstance(config, ConfigObject):
        raise TypeError(f"Config needs to be of type 'ConfigObject' not {type(config)}")

    batch_size = config.batch_size
    

    conv1 = leakyRelu(conv2d(inp, [1, 3, 1, 20], "conv1", padding="VALID"))
    print(conv1.shape)
    conv2 = leakyRelu(conv3d(conv1, [1, 20, 10, 20, 20], "conv2", padding="VALID"))
    print(conv2.shape)
    # conv2 = leakyRelu(conv2d(inp, [1, 13, 10], "conv1", padding="VALID"))

 
if __name__ == "__main__":

    config = ConfigObject(batch_size=64)
    
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y_ = tf.placeholder(tf.float32, [None, 2])

    network = CNN_classifier([22,2,2], config, 2)
