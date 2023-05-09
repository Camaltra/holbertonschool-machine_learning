#!/usr/bin/env python3


"""Useless comment"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    Function to seve the model
    :param network: The model to save
    :param filename: The path to save the model to
    :return: Nothing
    """
    network.save(filename)


def load_model(filename):
    """
    Load a model from a filepath
    :param filename: The filepath
    :return: The loaded model
    """
    return K.models.load_model(filename)
