#!/usr/bin/env python3

"""Useless comments"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Create a one hot encoded matrix from a 1D array
    :param labels: The 1D array
    :param classes: The number of classes
    :return: The one hot encoded matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)
