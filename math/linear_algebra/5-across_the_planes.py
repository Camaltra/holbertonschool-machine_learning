#!/usr/bin/env python3


"""Useless comment"""

matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(
        first_matrix: list | None,
        second_matrix: list | None,
) -> list | None:
    """
    Add two matrices
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :return: The sum of the matrices or None if not exist or not the same size
    """
    if first_matrix is None or second_matrix is None:
        return None
    if matrix_shape(first_matrix) != matrix_shape(second_matrix):
        return None
    return [[first_matrix_elem + second_matrix_elem for
             first_matrix_elem, second_matrix_elem in
             zip(first_matrix_row, second_matrix_row)] for
            first_matrix_row, second_matrix_row in
            zip(first_matrix, second_matrix)]
