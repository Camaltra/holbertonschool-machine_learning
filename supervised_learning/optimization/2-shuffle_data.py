#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle a dataset (X and Y set in the same ways)
    :param X: The dataset feature
    :param Y: The labels
    :return: The shuffled datasets
    """
    dataset_len = X.shape[0]
    indices_permutted = np.random.permutation(dataset_len)
    return X[indices_permutted], Y[indices_permutted]
