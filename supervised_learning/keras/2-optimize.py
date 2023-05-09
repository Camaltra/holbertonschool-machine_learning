#!/usr/bin/env python3

"""Useless comments"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta_1, beta_2):
    """
    Optimize a model using the Adam optimizer
    :param network: The model to optimize
    :param alpha: The learning rate
    :param beta_1: The first moment paramter
    :param beta_2: The second moment parameter
    :return: Nothing
    """
    optimizer = K.optimizers.legacy.Adam(
        alpha, beta_1=beta_1, beta_2=beta_2
    )
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
