#!/usr/bin/env python3


"""Useless comment"""


def cat_matrices2D(
        first_matrix: list,
        second_matrix: list,
        axis: int = 0,
) -> list | None:
    """
    Return the concat of two matrix depend on specific axis.
    Supposed that the matrices are not None
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :param axis: The axis of the concat
    :return: The newlly created matrix
    """
    if axis == 0:
        if len(next(iter(first_matrix), [])) != \
                len(next(iter(second_matrix), [])):
            return None
        return [[elem for elem in row] for row in first_matrix + second_matrix]
    else:
        if len(first_matrix) != len(second_matrix):
            return None
        return [[elem for elem in first_matrix_row + second_matrix_row] for
                first_matrix_row, second_matrix_row in
                zip(first_matrix, second_matrix)]
