#!/usr/bin/env python3


"""Useless comment"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def mat_mul(
        first_matrix: list[list],
        second_matrix: list[list],
) -> list[list] | None:
    """
    Multiply tzo python matrix
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :return: The mul of the python array
    """
    if matrix_shape(first_matrix)[-1] != matrix_shape(second_matrix)[0]:
        return None
    result_row_dim = len(first_matrix)
    result_col_dim = len(second_matrix[0])
    return [[sum(first_matrix[i][k] * second_matrix[k][j]
                 for k in range(len(second_matrix)))
             for j in range(result_col_dim)] for i in range(result_row_dim)]
