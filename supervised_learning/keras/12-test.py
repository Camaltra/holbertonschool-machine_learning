#!/usr/bin/env python3


"""Useless comment"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Test a model based on a given dataset
    :param network: The model to test
    :param data: The dataset features
    :param labels: The dataset labels
    :param verbose: The amount of verbose
    :return: The loss and accuracy of the model
    """
    return network.evaluate(data, labels, verbose=verbose)
