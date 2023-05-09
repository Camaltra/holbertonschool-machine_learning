#!/usr/bin/env python3

"""Useless comments"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """
    Train a model using mini-batch gradient descente
    :param network: The model to perform the fit on
    :param data: The dataset
    :param labels: The thruth labels
    :param batch_size: The batch size
    :param epochs: The number of epoch to perform
    :param validation_data: The validation dataset (Could not exist)
    :param early_stopping: If we want an early stopping
    :param patience: The patience in case of early stop
    :param learning_rate_decay: The learning rate decay used for the
                                 callback
    :param alpha: The initial learning_rate
    :param decay_rate: The decay rate
    :param verbose: The level of verbose (True or False)
    :param shuffle: If the data need to be shuffle before each epoach
    :return: The fited model
    """
    def lr_decay(epochs):
        return alpha / (1 + decay_rate * epochs)

    callback_list = []
    if learning_rate_decay and validation_data:
        callback_list.append(K.callbacks.LearningRateScheduler(lr_decay,
                                                               verbose=True))
    if early_stopping and validation_data:
        callback_list.append(K.callbacks.EarlyStopping(patience=patience))
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callback_list
    )
