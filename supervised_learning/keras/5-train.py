#!/usr/bin/env python3

"""Useless comments"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Train a model using mini-batch gradient descente
    :param network: The model to perform the fit on
    :param data: The dataset
    :param labels: The thruth labels
    :param batch_size: The batch size
    :param epochs: The number of epoch to perform
    :param validation_data: The validation dataset (Could not exist)
    :param verbose: The level of verbose (True or False)
    :param shuffle: If the data need to be shuffle before each epoach
    :return: The fited model
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
