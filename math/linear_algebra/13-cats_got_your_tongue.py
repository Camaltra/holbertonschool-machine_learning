#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def np_cat(
        first_matrix: np.ndarray,
        second_matrix: np.ndarray,
        axis: int = 0
) -> np.ndarray:
    """
    Concat two np array along an axis
    :param first_matrix: The fisrt np array
    :param second_matrix: The second np array
    :param axis: The axis to perform the concat
    :return: The nez np array
    """
    return np.concatenate((first_matrix, second_matrix), axis=axis)
