import numpy as np
import tensorflow as tf


weight_decay = 1e-4


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, is_train=True, name='bn'):
    return tf.layers.batch_normalization(x, momentum=momentum,
                                         epsilon=epsilon, scale=True,
                                         training=is_train, name=name)


def conv2d(input, output_dim, kernel_size, strides, stddev=0.02, use_bias=False, name='conv2d'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [kernel_size, kernel_size, input.get_shape()[-1], output_dim],
                                 regularizer=tf.contrib.layers.l2_regularizer(
                                     weight_decay),
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, weight, strides=[
                            1, strides, strides, 1], padding='SAME')
        if use_bias:
            bias = tf.get_variable(
                'bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        return conv


def conv2d_block(input, output_dim, kernel_size, strides, is_train, name, use_bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, output_dim, kernel_size, strides,
                     use_bias=use_bias, name='conv2d')
        net = batch_norm(net, is_train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, use_bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1, 1, use_bias=use_bias, name=name)


def point_wise(input, output_dim, is_train, name, use_relu=True, use_bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        pw = conv2d(input, output_dim, 1, 1,
                    use_bias=use_bias, name='conv_1x1')
        pw = batch_norm(pw, is_train=is_train, name='bn')
        if use_relu:
            pw = relu(pw)
        return pw


def depth_wise(input, kernel_size=3, strides=1, padding='SAME', channel_multiplier=1, is_train=True, use_bias=False, name='depth_wise'):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        weight = tf.get_variable('weight', [kernel_size, kernel_size, in_channel, channel_multiplier],
                                 regularizer=tf.contrib.layers.l2_regularizer(
                                     weight_decay),
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        dw = tf.nn.depthwise_conv2d(input, weight, strides=[
                                    1, strides, strides, 1], padding=padding, rate=None, name=None, data_format=None)
        if use_bias:
            bias = tf.get_variable(
                'bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            dw = tf.nn.bias_add(dw, bias)
        dw = batch_norm(dw, is_train=is_train, name='bn')
        dw = relu(dw)
        return dw


def bottleneck(input, expansion_ratio, output_dim, strides, is_train, name, use_bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        bottleneck_dim = round(
            expansion_ratio*input.get_shape().as_list()[-1])
        net = point_wise(input, bottleneck_dim, is_train,
                         use_bias=use_bias, name='point_wise_1')
        net = depth_wise(net, strides=strides,
                         is_train=is_train, use_bias=use_bias)
        net = point_wise(net, output_dim, is_train, use_relu=False,
                         use_bias=use_bias, name='point_wise_2')
        if shortcut and strides == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1x1(input, output_dim, name='ex_dim')
                net = ins+net
            else:
                net = input+net
        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                                       regularizer=tf.contrib.layers.l2_regularizer(
                                           weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                                       regularizer=tf.contrib.layers.l2_regularizer(
                                           weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1, stride, stride, 1]

        conv = tf.nn.separable_conv2d(
            input, dwise_filter, pwise_filter, strides, padding=pad, name=name)
        if bias:
            biases = tf.get_variable(
                'bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    # flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net


__all__ = ['conv2d_block', 'point_wise', 'bottleneck', 'global_avg', 'flatten']
