#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def np_elementwise(
        first_matrix: np.ndarray,
        second_matrix: np.ndarray
) -> tuple[np.ndarray]:
    """
    Perfom different operation through numpy
    :param first_matrix: The fist matrix
    :param second_matrix: The second matrix
    :return: All the perfomer operation
    """
    return (np.add(first_matrix, second_matrix),
            np.subtract(first_matrix, second_matrix),
            np.multiply(first_matrix, second_matrix),
            np.divide(first_matrix, second_matrix))
