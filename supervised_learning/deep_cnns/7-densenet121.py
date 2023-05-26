#!/usr/bin/env python3

"""Useless comment"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Create the DenseNet121-bc model
    :param growth_rate: The growth rate
    :param compression: The compression rate
    :return:
    """
    init = K.initializers.he_normal()
    X = K.layers.Input(shape=(224, 224, 3))
    norm_1 = K.layers.BatchNormalization()(X)
    act_1 = K.layers.Activation('relu')(norm_1)
    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding="same",
                             kernel_initializer=init)(act_1)
    pool_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                   strides=(2, 2),
                                   padding="same")(conv_1)

    dense_1, nb_filters = dense_block(pool_1, 64, growth_rate, 6)
    trans_1, nb_filters = transition_layer(dense_1, nb_filters, compression)

    dense_2, nb_filters = dense_block(trans_1, nb_filters, growth_rate, 12)
    trans_2, nb_filters = transition_layer(dense_2, nb_filters, compression)

    dense_3, nb_filters = dense_block(trans_2, nb_filters, growth_rate, 24)
    trans_3, nb_filters = transition_layer(dense_3, nb_filters, compression)

    dense_4, nb_filters = dense_block(trans_3, nb_filters, growth_rate, 16)

    pool_2 = K.layers.AveragePooling2D(pool_size=(7, 7))(dense_4)
    fc_1 = K.layers.Dense(units=1000,
                          activation="softmax",
                          kernel_initializer=init)(pool_2)

    model = K.models.Model(inputs=X, outputs=fc_1)
    return model
