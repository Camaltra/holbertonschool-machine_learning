#!/usr/bin/env python3


"""useless comment"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Compute the expectation (ie posterior probs and likelihood)
    :param X: The dataset
    :param pi: The prior probs
    :param m: The means
    :param S: The covariance matrix
    :return: The posterior probs and the likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if m.shape[1] != X.shape[1]:
        return None, None
    if S.shape[1] != X.shape[1] or X.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    num_of_clust = len(pi)
    probas = np.zeros(shape=(num_of_clust, X.shape[0]))
    for i in range(num_of_clust):
        probas[i] = pdf(X, m[i], S[i]) * pi[i]

    maginal = np.sum(probas, axis=0)
    probas = probas / maginal
    max_likelihood = np.sum(np.log(maginal))

    return probas, max_likelihood
