#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def absorbing(P):
    """
    Check is a markov chain is absorbing
    :param P: The transition matrix
    :return: True or False
    """
    m, n = P.shape
    diag = np.diagonal(P)

    if np.all(diag == 1):
        return True
    if np.all(diag != 1):
        return False

    for i in range(m):
        if not np.any(P[i, :] == 1):
            break

    Q = P[i:, i:]

    F_inv = np.eye(Q.shape[0]) - Q
    if np.linalg.det(F_inv) == 0:
        return False
    return True
