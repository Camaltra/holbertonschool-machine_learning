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
    U, S, Vt = np.linalg.svd(X)
    return np.matmul(U[..., :ndim], np.diag(S.T[..., :ndim]))
