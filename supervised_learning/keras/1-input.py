#!/usr/bin/env python3

"""Useless comments"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a simple deep learning model using L2 regularisation and
    droupout layers
    :param nx: The number of features of the dataset
    :param layers: An array that cointain how much node a layer [i] got
    :param activations: The activation functions for each layers at index [i]
    :param lambtha: The L2 parameter
    :param keep_prob: The prob for the dropout layer
    :return: The created model
    """
    layers_sequence = []
    regularize_l2 = K.regularizers.l2(l=lambtha)
    for index, (layer, activation) in enumerate(zip(layers, activations)):
        layers_sequence.append(
            K.layers.Dense(
                layer,
                activation=activation,
                kernel_regularizer=regularize_l2
            )
        )
        if index == len(layers) - 1:
            continue
        layers_sequence.append(K.layers.Dropout(1 - keep_prob))

    inputs = K.Input(shape=(nx,))
    outputs = inputs
    for layer in layers_sequence:
        outputs = layer(outputs)
    return K.Model(inputs=inputs, outputs=outputs)
