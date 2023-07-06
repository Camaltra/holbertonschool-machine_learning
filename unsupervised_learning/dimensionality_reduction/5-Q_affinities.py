#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def Q_affinities(Y):
    """
    Compute the Q affinities
    :param Y: Containing the low dimensional transformation of X
    :return: The Q affinities and the numerators of the Q affinities
    """
    D = (np.sum(Y**2, axis=1) + np.sum(Y**2, axis=1)[..., np.newaxis]
         - 2 * np.dot(Y, Y.T))
    np.fill_diagonal(D, 0)

    numerator = 1 / (1 + D)
    np.fill_diagonal(numerator, 0)
    Q = numerator / np.sum(numerator)

    return Q, numerator
