#!/usr/bin/env python3

"""Useless comment"""
import tensorflow as tf


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Create the dense block of the DenseNet-b
    :param X: The output of the previous layer
    :param nb_filters: The number of filter from the previous
    :param growth_rate: The growth rate
    :param layers: The number of layer in the dense block
    :return: The dense block module
    """
    init = tf.keras.initializers.he_normal()
    input_data = X
    for _ in range(layers):
        norm = tf.keras.layers.BatchNormalization()(input_data)
        act = tf.keras.layers.ReLU()(norm)
        conv_1x1 = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                          kernel_size=(1, 1),
                                          kernel_initializer=init)(act)
        norm = tf.keras.layers.BatchNormalization()(conv_1x1)
        act = tf.keras.layers.ReLU()(norm)
        conv_3x3 = tf.keras.layers.Conv2D(filters=growth_rate,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=init)(act)
        input_data = tf.keras.layers.Concatenate()([input_data, conv_3x3])

    return input_data, growth_rate * nb_filters
