#!/usr/bin/env python3


"""useless comment"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    a function that creates a tensrflow layer with L2 regularization
    :param prev: is a tensor containing the output of the previous layer
    :param n: the number of nodes the new layer should contain
    :param activation: the activation function that should be used on the layer
    :param keep_prob: the proba of keeping some neuron on certain layer
    :return: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer", kernel_regularizer=reg)
    return layer(prev)
