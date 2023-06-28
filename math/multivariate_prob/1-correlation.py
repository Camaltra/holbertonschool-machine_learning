#!/usr/bin/env python3

"""useless comment"""

import numpy as np


def correlation(C):
    """

    :param C:
    :return:
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if not len(C.shape) == 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    diag = np.diag(1 / np.sqrt(np.diag(C)))
    return np.matmul(np.matmul(diag, C), diag)
