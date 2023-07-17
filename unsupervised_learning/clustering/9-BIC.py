#!/usr/bin/env python3


"""useless comment"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """calculates BIC over various """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if type(kmin) is not int or kmin != int(kmin) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if type(kmax) is not int or kmax != int(kmax) or kmax < 1:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations != int(iterations) or iterations < 1:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None
    b = np.zeros(kmax + 1 - kmin)
    l = np.zeros(kmax + 1 - kmin)
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, _, l[k - kmin] = expectation_maximization(X, k, iterations=iterations, tol=tol, verbose=verbose)
        results.append((pi, m, S))
        p = k * (d + 2) * (d + 1) / 2 - 1
        b[k - kmin] = p * np.log(n) - 2 * l[k - kmin]
    amin = np.argmin(b)
    best_k = amin + kmin
    best_result = results[amin]
    return best_k, best_result, l, b
