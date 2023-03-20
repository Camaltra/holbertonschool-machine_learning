#!/usr/bin/env python3

"""Useless comments"""


def matrix_transpose(matrix):
    """
    Transpose a matrix
    :param matrix: A given Matrix
    :return: The newlly transpose matrix or a empty list if the
             given matrix is empty
    """
    if matrix is None:
        return []
    return [[matrix[col_idx][row_idx] for
             col_idx in range(len(matrix))] for
            row_idx in range(len(matrix[0]))]
