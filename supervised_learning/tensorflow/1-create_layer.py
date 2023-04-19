#!/usr/bin/env python3

"""useless comments"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Create placeholders tensor
    :param nx: The number of feature columns in our data
    :param classes: The number of classes in our classifier
    :return: The two placeholders
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
