#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def cost(P, Q):
    """
    Compute the cost
    :param P: All P affinities
    :param Q: All Q affinities
    :return: The cost for a given step
    """
    return (np.sum(np.maximum(P, 1e-12) * np.log(np.maximum(P, 1e-12)
                                                 / (np.maximum(Q, 1e-12)))))
