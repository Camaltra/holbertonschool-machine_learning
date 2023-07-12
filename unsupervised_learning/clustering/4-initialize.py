#!/usr/bin/env python3


"""useless comment"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initilize variable (pi, means, covariance matrix)
    for a Gaussian Mixture Model
    :param X: The dataset
    :param k: The number of cluster
    :return: The prior probs, means and covariance matrix
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    _, d = X.shape

    pi = np.ones(shape=(k,)) / k
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S
