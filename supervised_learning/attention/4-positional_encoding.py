#!/usr/bin/env python3

"""useless comment"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Compute the positionnal encoding
    :param max_seq_len: The max_seq_len
    :param dm: The model depth
    :return: The positionnal encoding
    """
    P = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm / 2)):
            denominator = np.power(10000, 2 * i / dm)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P
