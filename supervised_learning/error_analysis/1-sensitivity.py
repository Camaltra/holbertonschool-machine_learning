#!/usr/bin/env python3

"""useless comments"""

import numpy as np


def sensitivity(confusion):
    """
    Compute the sensitivity for all classes
    :param confusion: The confisuion matrix
    :return: The sensitivity for all classes
    """
    n, _ = confusion.shape
    sensitivity_matrix = np.empty((n,))
    for idx in range(n):
        sensitivity_matrix[idx] = confusion[idx][idx] / np.sum(confusion[idx])
    return sensitivity_matrix
