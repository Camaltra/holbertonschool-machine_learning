#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def mean_cov(X):
    """
    Copmpute the mean and the covariance matrix
    :param X: The given matrix
    :return: Mean and covariance matrix
    """
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    cols, n = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    means = np.mean(X, axis=1).reshape(cols, 1)
    X_mean = X - means
    cov_matrix = np.matmul(X_mean, X_mean.T) / (n - 1)

    return means, np.array(cov_matrix)


class MultiNormal:
    """Comments"""
    def __init__(self, data):
        self.data = data
        self.mean, self.cov = mean_cov(self.data)

    def pdf(self, x):
        k = self.mean.shape[0]
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (k, 1):
            raise ValueError("x must have the shape ({}, 1)".format(k))

        coef = 1 / ((2 * np.pi)**(k / 2) * np.sqrt(np.linalg.det(self.cov)))
        exponetial_arg = (np.dot(
            np.dot(
                (x - self.mean).T,
                np.linalg.inv(self.cov)
            ),
            (x - self.mean)
        )) / - 2

        return (coef * np.exp(exponetial_arg))[0][0]
