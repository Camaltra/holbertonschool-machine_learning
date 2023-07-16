#!/usr/bin/env python3


"""useless comment"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Use the BIC algos to find the best K parameter the EM algo
    :param X: The dataset
    :param kmin: kmin
    :param kmax: kmax
    :param iterations: The number of iteration the EM algo
    :param tol: The tol the EM algo
    :param verbose: The verbose the EM aglo
    :return: Best K, Best res and the history of both l and b
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) != int or kmax < 1:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) != int or iterations < 1:
        return None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    n, d = X.shape
    k_values_list = []
    results_list = []
    log_likelihood_list = []
    bic_value_list = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        p = d * k + (d * k * (d + 1) / 2) + k - 1
        BIC = p * np.log(n) - 2 * log_likelihood

        k_values_list.append(k)
        results_list.append((pi, m, S))
        log_likelihood_list.append(log_likelihood)
        bic_value_list.append(BIC)

        log_likelihood_array = np.array(log_likelihood_list)
        bic_value_array = np.array(bic_value_list)
        index = np.argmin(bic_value_array)

        best_k = k_values_list[index]
        best_result = results_list[index]

    return best_k, best_result, log_likelihood_array, bic_value_array
