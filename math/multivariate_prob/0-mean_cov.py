#!/usr/bin/env python3

"""useless comment"""

import numpy as np


def mean_cov(X):
    """
    Copmpute the mean and the covariance matrix
    :param X: The given matrix
    :return: Mean and covariance matrix
    """
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, cols = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    means = np.mean(X, axis=0).reshape(1, cols)
    X_mean = X - means
    cov_matrix = np.matmul(X_mean.T, X_mean) / (n - 1)

    return means, np.array(cov_matrix)
