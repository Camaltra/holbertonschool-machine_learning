#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Encode a list
    :param Y: list
    :param classes: The max value of the list
    :return: The encoded matrix
    """
    one_hot_matrix = np.zeros((Y.size, classes))
    one_hot_matrix[np.arange(Y.size), Y] = 1
    return one_hot_matrix
