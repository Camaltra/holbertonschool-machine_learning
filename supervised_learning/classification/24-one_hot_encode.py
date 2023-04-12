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
    if not isinstance(Y, np.ndarray):
        return None
    if classes <= 2 or classes > len(Y):
        return None
    one_hot_matrix = np.zeros((Y.size, classes))
    one_hot_matrix[np.arange(Y.size), Y] = 1
    return one_hot_matrix.T
