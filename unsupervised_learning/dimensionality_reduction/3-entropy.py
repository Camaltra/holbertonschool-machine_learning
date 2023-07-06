#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def HP(Di, beta):
    """
    Compute the Shannon entropy and P affinities relative to a data point
    :param Di: The pariwise distances between a
               data point and all other points except itself
    :param beta: The beta value for the Gaussian distribution
    :return: The Shannon entropy and P affinities
    """
    Di_tmp = np.exp(-Di * beta)
    Di_sum = np.sum(Di_tmp)

    Pi = Di_tmp / Di_sum

    H_pi = - np.sum(Pi * np.log2(Pi))

    return H_pi, Pi
