#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def pca(X, ndim):
    """
    Compute the PCA
    :param X: The base dataset
    :param ndim: The number of dims
    :return: PCA
    """
    X = X - np.mean(X, axis=0)
    _, __, Vt = np.linalg.svd(X)
    return np.matmul(X, Vt.T[..., :ndim])
