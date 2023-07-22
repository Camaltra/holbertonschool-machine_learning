#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def regular(P):
    """
    Compute the steady state probabilities
    :param P: The transition matrix
    :return: The steady state probabilities
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if np.any(P <= 0):
        return None
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    if eigenvalues is None or eigenvectors is None:
        return None, None

    index = np.where(np.isclose(eigenvalues, 1))
    if len(index) < 1:
        return None, None
    index = index[0][0]

    steady_state_probabilities = np.real(eigenvectors[:, index].T)

    return (
            steady_state_probabilities / np.sum(steady_state_probabilities)
    ).reshape(1, n)
