#!/usr/bin/env python3


"""useless comments"""


import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, Pij):
    """
    Compute the grads
    :param Y: Containing the low dimensional transformation of X
    :param Pij: The Pij for all ij affinities
    :return: The grads
    """
    dY = np.zeros(Y.shape)

    # dY = (2500, 2)
    Qij, numerator = Q_affinities(Y)

    PQij = Pij - Qij

    for i in range(Y.shape[0]):
        dY[i, :] = np.dot((PQij[i, :] * numerator[i, :]).T, (Y - Y[i, :]))

    return dY, Qij
