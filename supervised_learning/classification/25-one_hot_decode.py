#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def one_hot_decode(Y):
    """
    Decode a one hot encoded matrix
    :param Y: The encoded matrix
    :return: The decoded matrix
    """
    decoded_list = []
    for row in Y.T:
        decoded_list.append(row.argmax())
    return np.array(decoded_list)
