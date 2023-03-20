#!/usr/bin/env python3


"""Useless comment"""


def np_elementwise(first_matrix, second_matrix) -> tuple:
    """
    Perfom different operation through numpy
    :param first_matrix: The fist matrix
    :param second_matrix: The second matrix
    :return: All the perfomer operation
    """
    return (first_matrix + second_matrix,
            first_matrix - second_matrix,
            first_matrix * second_matrix,
            first_matrix / second_matrix)
