#!/usr/bin/env python3

"""Useless comment"""


import tensorflow.keras as K


def lenet5(X):
    """
    That builds a modified version of the LeNet-5
    architecture using tensorflow
    :param X: Is a `tf.placeholder` of shape (m, 28, 28, 1)
              containing the input images for the network
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

    prev_layer_output = X
    for layer in layers:
        prev_layer_output = layer(prev_layer_output)

    model = K.Model(inputs=X, outputs=prev_layer_output)

    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
