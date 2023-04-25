#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a layer that normalized the unactivate input data
    :param prev: The prev layers ouput
    :param n: The number of node in the layer
    :param activation: The activation fuction
    :return: The new created layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense_layer = tf.layers.Dense(units=n,
                                  kernel_initializer=init)

    mean, variance = tf.nn.moments(prev, axes=[0])

    scale = tf.Variable(tf.ones([n]))
    shift = tf.Variable(tf.zeros([n]))

    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(prev, mean, variance,
                                           shift, scale, epsilon)

    output = normalized(prev)

    if activation is not None:
        output = activation(output)

    return output
