#!/usr/bin/env python3

"""Useless comment"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Create the transision layer of the DenseNet-c model
    :param X: The output of the previous filter
    :param nb_filters: The number of filter of the previous layer
    :param compression: The compression rate
    :return:
    """
    init = K.initializers.he_normal()
    norm = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(norm)
    conv_1x1 = K.layers.Conv2D(filters=int(nb_filters*compression),
                               kernel_size=(1, 1),
                               kernel_initializer=init)(act)
    pooling = K.layers.MaxPooling2D(pool_size=(2, 2),
                                    padding="same",
                                    strides=(1, 1))(conv_1x1)

    return pooling, int(nb_filters*compression)
