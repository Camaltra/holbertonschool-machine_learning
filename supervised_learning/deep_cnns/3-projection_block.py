#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf


def projection_block(A_prev, filters, s=2):
    """
    Create the projection block of the ResNet Model
    :param A_prev: The previous layer output
    :param filters: An array of the conv-filter-size
    :param s: The stride of the shortcut
    :return: The projection block
    """

    init = tf.keras.initializers.he_normal()
    f11, f3, f12 = filters

    conv_f12_shortcut = tf.keras.layers.Conv2D(filters=f12,
                                               kernel_size=(1, 1),
                                               padding="same",
                                               kernel_initializer=init,
                                               strides=(s, s))(A_prev)
    norm_1_shortcut = tf.keras.layers.BatchNormalization()(conv_f12_shortcut)

    conv_f11 = tf.keras.layers.Conv2D(filters=f11,
                                      kernel_size=(1, 1),
                                      padding="same",
                                      kernel_initializer=init,
                                      strides=(s, s))(A_prev)
    norm_1 = tf.keras.layers.BatchNormalization()(conv_f11)
    act_1 = tf.keras.layers.ReLU()(norm_1)

    conv_f3 = tf.keras.layers.Conv2D(filters=f3,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=init)(act_1)
    norm_2 = tf.keras.layers.BatchNormalization(axis=3)(conv_f3)
    act_2 = tf.keras.layers.ReLU()(norm_2)

    conv_f12 = tf.keras.layers.Conv2D(filters=f12,
                                      kernel_size=(1, 1),
                                      padding="same",
                                      kernel_initializer=init)(act_2)
    norm_3 = tf.keras.layers.BatchNormalization(axis=3)(conv_f12)

    add = tf.keras.layers.Add()([norm_3, norm_1_shortcut])

    return tf.keras.layers.ReLU()(add)
