#!/usr/bin/env python3

"""Useless Comment"""


def matrix_shape(matrix: list | None) -> list:
    """
    Get the shape of a given matrix
    :param matrix: A python list
    :return: A list that contain the dimension of the given matrix
    """
    if matrix is None:
        return []
    matrix_dim = []
    while isinstance(matrix, list):
        matrix_dim.append(len(matrix))
        matrix = next(iter(matrix), None)
    return matrix_dim
