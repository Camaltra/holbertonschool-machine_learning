#!/usr/bin/env python3


"""Useless comment"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Save the weights of a model
    :param network: The model to save the weights
    :param filename: The filepath to save the model to
    :param save_format: The save format
    :return: Noting
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load weights form a file
    :param network: The model to load the weights to
    :param filename: The filepath to load the weights
    :return: Nothing
    """
    network.load_weights(filename)
