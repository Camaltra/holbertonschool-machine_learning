#!/usr/bin/env python3

"""useless comment"""

import numpy as np


def correlation(C):
    """

    :param C:
    :return:
    """
    diag = np.diag(1 / np.sqrt(np.diag(C)))
    return np.matmul(np.matmul(diag, C), diag)
