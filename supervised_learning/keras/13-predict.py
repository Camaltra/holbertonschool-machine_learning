#!/usr/bin/env python3


"""Useless comment"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Make prediction on a given dataset using a
    given model
    :param network: The givne model
    :param data: The data
    :param verbose: The amount of verbose
    :return: The predictions
    """
    return network.predict(data, verbose=verbose)