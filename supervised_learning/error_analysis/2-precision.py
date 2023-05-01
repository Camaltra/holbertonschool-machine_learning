#!/usr/bin/env python3

"""useless comments"""

import numpy as np


def precision(confusion):
    """
    Compute the precision for all classes
    :param confusion: The confisuion matrix
    :return: The precision for all classes
    """
    n, _ = confusion.shape
    precision_matrix = np.empty((n,))
    for idx in range(n):
        precision_matrix[idx] = confusion[idx][idx] / np.sum(confusion[:, idx])
    return precision_matrix
