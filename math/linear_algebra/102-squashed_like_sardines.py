#!/usr/bin/env python3


"""Useless comment"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def _get_matrix_dim(matrix) -> int:
    """
    Get the dimension of a given matrix
    Suppose that the matrix exist
    :param matrix: The given matrix
    :return: The numbers of axes of the matrix
    """
    return len(matrix_shape(matrix))


def _is_concatenable(first_matrix, second_matrix, axis: int) -> bool:
    """
    Check is two matrices can be concatante
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :param axis: The axis on where we want perform the concat
    :return: True or false
    """
    first_matrix_dim = _get_matrix_dim(first_matrix)
    second_matrix_dim = _get_matrix_dim(second_matrix)
    first_matrix_shape = matrix_shape(first_matrix)
    second_matrix_shape = matrix_shape(second_matrix)
    if first_matrix_dim != second_matrix_dim:
        return False
    if min(first_matrix_dim, second_matrix_dim) < axis + 1:
        return False
    for i in range(first_matrix_dim):
        if i == axis:
            continue
        if first_matrix_shape[i] != second_matrix_shape[i]:
            return False
    return True


def _cat_matrices(first_matrix, second_matrix, axis: int):
    """
    Recusive function that concat two given matrices
    No check are perform on matrices here
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :param axis: The axis we curently working on
    :return: The concat matrix result
    """
    output_matrix = []
    if axis == 0:
        return first_matrix + second_matrix
    else:
        for i in range(len(first_matrix)):
            output_matrix.append(
                _cat_matrices(first_matrix[i], second_matrix[i], axis - 1)
            )
    return output_matrix


def cat_matrices(first_matrix, second_matrix, axis: int = 0):
    """
    Main interface function for concat 2 matrices
    :param first_matrix: The first matrix
    :param second_matrix: The second matrix
    :param axis: The axis on where we want perform the concat
    :return: The concat matrix
    """
    if not _is_concatenable(first_matrix, second_matrix, axis):
        return None
    return _cat_matrices(first_matrix, second_matrix, axis)
