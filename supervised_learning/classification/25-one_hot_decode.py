#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def one_hot_decode(Y):
    """
    Decode a one hot encoded matrix
    :param Y: The encoded matrix
    :return: The decoded matrix
    """
    if not isinstance(Y, np.ndarray):
        return None
    if len(Y) == 0 or len(Y.shape) != 2:
        return None
    return np.argmax(Y, axis=0)
