#!/usr/bin/env python3


"""Useless comment"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    Save the model configuration
    :param network: The model to save its configuration
    :param filename: The filename
    :return: None
    """
    network_json = network.to_json()
    with open(filename, "w") as f:
        f.write(network_json)


def load_config(filename):
    """
    Load a model configuration from a json file
    :param filename: The filename
    :return: The loaded model
    """
    with open(filename, "r") as f:
        loaded_network_json = f.read()
    return K.models.model_from_json(loaded_network_json)
