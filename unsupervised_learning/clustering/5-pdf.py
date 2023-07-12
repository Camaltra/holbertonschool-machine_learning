#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def pdf(X, m, S):
    """
    Compute the pdf for a Gaussian Mixture Model
    :param X: The dataset
    :param m: The means
    :param S: Teh covarance matrix
    :return: The pdf
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if m.shape[0] != X.shape[1]:
        return None
    if S.shape[0] != X.shape[1]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    coef = 1 / ((2 * np.pi)**(d / 2) * np.sqrt(np.linalg.det(S)))
    exponetial_arg = np.sum(
        np.dot(
            (X - m),
            np.linalg.inv(S)
        ) * (X - m),
        axis=1
    ) / - 2

    return np.maximum(coef * np.exp(exponetial_arg), 1e-300)
