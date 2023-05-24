#!/usr/bin/env python3

"""Useless comment"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Create the identity block form the ResNet Model
    :param A_prev: The previous layer output
    :param filters: An array of the conv-filter-size
    :return: The identity module
    """
    init = K.initializers.he_normal()
    f11, f3, f12 = filters

    conv_f11 = K.layers.Conv2D(filters=f11,
                                      kernel_size=(1, 1),
                                      padding="same",
                                      kernel_initializer=init)(A_prev)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_f11)
    act_1 = K.layers.Activation('relu')(norm_1)

    conv_f3 = K.layers.Conv2D(filters=f3,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=init)(act_1)
    norm_2 = K.layers.BatchNormalization(axis=3)(conv_f3)
    act_2 = K.layers.Activation('relu')(norm_2)

    conv_f12 = K.layers.Conv2D(filters=f12,
                                      kernel_size=(1, 1),
                                      padding="same",
                                      kernel_initializer=init)(act_2)
    norm_3 = K.layers.BatchNormalization(axis=3)(conv_f12)

    add = K.layers.Add()([norm_3, A_prev])

    return K.layers.Activation('relu')(add)
