#!/usr/bin/env python3


"""useless comment"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Compute the Kmean for different values of k on a given dataset
    Output the variance and the values for K.
    :param X: The dataset
    :param kmin: The value min for K
    :param kmax: The value max for K
    :param iterations: The number of iterations for the Kmean algos
    :return: The variance and the values for K
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, _ = X.shape
    kmax = kmax if kmax is not None else n
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if kmin <= 0 or kmax <= 0 or kmax <= kmin:
        return None, None
    if not isinstance(iterations, np.int) or iterations <= 0:
        return None, None

    kmean_results = []
    var_results = []
    base_var = 0
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        kmean_results.append((C, clss))

        var = variance(X, C)
        if not len(var_results):
            base_var = var
        var_results.append(base_var - var)

    return kmean_results, var_results
