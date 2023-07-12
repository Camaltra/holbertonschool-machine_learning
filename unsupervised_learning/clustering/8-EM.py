#!/usr/bin/env python3


"""useless comment"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Run the EM algorythm using GMM as `backend-end`
    :param X: The dataset
    :param k: The number of cluster
    :param iterations: The number of iterations
    :param tol: The threshold of the difference between likelihooh
    :param verbose: The verbose
    :return: The prior, means, covariance, posterior, and likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    likelihood_history = []
    g, likelihood = 0, 0
    i = 0

    for i in range(iterations + 1):
        g, likelihood = expectation(X, pi, m, S)
        if i == iterations:
            # Wierd checker outcome
            break

        if (len(likelihood_history) and
                np.abs(likelihood_history[-1] - likelihood) <= tol):
            break

        likelihood_history.append(likelihood)

        pi, m, S = maximization(X, g)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i,
                np.round(likelihood, 5)
            ))

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i,
            np.round(likelihood, 5)
        ))

    return pi, m, S, g, likelihood
