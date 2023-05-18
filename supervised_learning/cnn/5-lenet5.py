#!/usr/bin/env python3

"""Useless comment"""


import tensorflow.keras as K


def lenet5(_):
    """
    That builds a modified version of the LeNet-5
    architecture using tensorflow
    :param _: Unsused variable
    :return: K.Model compiled to use Adam optimization
             (with default hyperparameters) and accuracy
             metrics
    """
    init = K.initializers.he_normal()

    layers = [
        K.layers.Conv2D(
            filters=6, activation="relu", kernel_size=(5, 5),
            padding="same", kernel_initializer=init,
        ),
        K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        K.layers.Conv2D(
            filters=16, activation="relu", kernel_size=(5, 5),
            padding="valid", kernel_initializer=init
        ),
        K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        K.layers.Flatten(),
        K.layers.Dense(120, activation="relu", kernel_initializer=init),
        K.layers.Dense(84, activation="relu", kernel_initializer=init),
        K.layers.Dense(10, activation="softmax", kernel_initializer=init),
    ]

    model = K.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
