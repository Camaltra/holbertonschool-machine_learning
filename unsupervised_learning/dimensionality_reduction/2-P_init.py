#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def P_init(X, perplexity):
    """
    Init the t-SNE variable:
    D -- The pairwise eucliedian distance between 2 points
    P -- Matrix of  P affinities
    b -- All of the beta values
    H -- Shannon entropy for perplexity with a base of 2
    :param X: The dataset
    :param perplexity: The perplexity value
    :return: D, P, b, H
    """
    n, _ = X.shape

    D = (np.sum(X**2, axis=1) + np.sum(X**2, axis=1)[..., np.newaxis]
         - 2 * np.dot(X, X.T))
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))

    b = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, b, H
