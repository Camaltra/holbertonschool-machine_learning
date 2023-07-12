#!/usr/bin/env python3


"""useless comment"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Use the BIC algos to find the best K parameter for the EM algo
    :param X: The dataset
    :param kmin: kmin
    :param kmax: kmax
    :param iterations: The number of iteration for the EM algo
    :param tol: The tol for the EM algo
    :param verbose: The verbose for the EM aglo
    :return: Best K, Best res and the history of both l and b
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    all_likelihood = []
    all_bic_value = []
    best_bic_value = 0
    n, d = X.shape

    for k in range(kmin, kmax + 1):
        pi, m, S, g, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        p = k * d + k * (d * (d + 1) / 2) + k - 1
        current_bic_value = p * np.log(n) - 2 * likelihood

        all_likelihood.append(likelihood)
        all_bic_value.append(current_bic_value)

        if current_bic_value > best_bic_value:
            best_k = k
            best_result = [pi, m, S]

    return best_k, best_result, all_likelihood, all_bic_value
