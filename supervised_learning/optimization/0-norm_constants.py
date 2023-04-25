#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def normalization_constants(X):
    """
    Compute the mean and the std for each feature
    :param X: The data set
    :return: The mean and the std for each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
