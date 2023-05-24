#!/usr/bin/env python3

"""Useless comment"""
import tensorflow as tf


def transition_layer(X, nb_filters, compression):
    """
    Create the transision layer of the DenseNet-c model
    :param X: The output of the previous filter
    :param nb_filters: The number of filter of the previous layer
    :param compression: The compression rate
    :return:
    """
    init = tf.keras.initializers.he_normal()
    norm = tf.keras.layers.BatchNormalization()(X)
    act = tf.keras.layers.ReLU()(norm)
    conv_1x1 = tf.keras.layers.Conv2D(filters=int(nb_filters*compression),
                                      kernel_size=(1, 1),
                                      kernel_initializer=init)(act)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding="same",
                                           strides=(1, 1))(conv_1x1)

    return pooling, int(tf.floor(nb_filters*compression))
