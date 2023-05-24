#!/usr/bin/env python3

"""Useless comment"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Create an inception block from the GoogleNet Model
    :param A_prev: The previous layer output
    :param filters: An array of the number of filter for each
                    sub-conv-layers
    :return: The output of the inception
    """
    f1, f3r, f3, f5r, f5, fpp = filters

    conv_f1 = K.layers.Conv2D(filters=f1,
                              kernel_size=(1, 1),
                              activation="relu")(A_prev)

    conv_f3r = K.layers.Conv2D(filters=f3r,
                               kernel_size=(1, 1),
                               activation="relu")(A_prev)
    conv_f3 = K.layers.Conv2D(filters=f3,
                              kernel_size=(3, 3),
                              padding="same",
                              activation="relu")(conv_f3r)

    conv_f5r = K.layers.Conv2D(filters=f5r,
                               kernel_size=(1, 1),
                               activation="relu")(A_prev)
    conv_f5 = K.layers.Conv2D(filters=f5,
                              kernel_size=(5, 5),
                              padding="same",
                              activation="relu")(conv_f5r)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding="same",
                                     strides=(1, 1))(A_prev)
    conv_fpp = K.layers.Conv2D(filters=fpp,
                               kernel_size=(1, 1),
                               activation="relu")(max_pool)

    return K.layers.Concatenate(axis=3)(
        [conv_f1, conv_f3, conv_f5, conv_fpp]
    )
