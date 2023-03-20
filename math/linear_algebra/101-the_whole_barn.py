#!/usr/bin/env python3


"""Useless comment"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def _add_matrices(first_matrix, second_matrix):
    """
    Internal function for adding two matrices
    :param first_matrix: The first python matrix
    :param second_matrix: The second python matrix
    :return: The sum of the two given python array
    """
    output_array = []
    for i in range(len(first_matrix)):
        if isinstance(first_matrix[i], list):
            sub_array = _add_matrices(first_matrix[i], second_matrix[i])
            output_array.append(sub_array)
        else:
            output_array.append(first_matrix[i] + second_matrix[i])
    return output_array


def add_matrices(first_matrix, second_matrix):
    """
    Adding two matrices
    :param first_matrix: The first python matrix
    :param second_matrix: The second python matrix
    :return: The sum of the two given python array
    """
    if matrix_shape(first_matrix) != matrix_shape(second_matrix):
        return None
    return _add_matrices(first_matrix, second_matrix)
