#!/usr/bin/env python3

"""Useless comment"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Create the projection block of the ResNet Model
    :param A_prev: The previous layer output
    :param filters: An array of the conv-filter-size
    :param s: The stride of the shortcut
    :return: The projection block
    """

    init = K.initializers.he_normal()
    f11, f3, f12 = filters

    conv_f11 = K.layers.Conv2D(filters=f11,
                               kernel_size=(1, 1),
                               padding="same",
                               kernel_initializer=init,
                               strides=(s, s))(A_prev)
    norm_1 = K.layers.BatchNormalization()(conv_f11)
    act_1 = K.layers.ReLU()(norm_1)

    conv_f3 = K.layers.Conv2D(filters=f3,
                              kernel_size=(3, 3),
                              padding="same",
                              kernel_initializer=init)(act_1)
    norm_2 = K.layers.BatchNormalization(axis=3)(conv_f3)
    act_2 = K.layers.ReLU()(norm_2)

    conv_f12 = K.layers.Conv2D(filters=f12,
                               kernel_size=(1, 1),
                               padding="same",
                               kernel_initializer=init)(act_2)
    norm_3 = K.layers.BatchNormalization(axis=3)(conv_f12)

    conv_f12_shortcut = K.layers.Conv2D(filters=f12,
                                        kernel_size=(1, 1),
                                        padding="same",
                                        kernel_initializer=init,
                                        strides=(s, s))(A_prev)
    norm_1_shortcut = K.layers.BatchNormalization()(conv_f12_shortcut)

    add = K.layers.Add()([norm_3, norm_1_shortcut])

    return K.layers.ReLU()(add)
