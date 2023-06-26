#!/usr/bin/env python3

"""useless comment"""


def determinant(matrix):
    """
    Compute the determinant of a matrix
    :param matrix: The given matrix
    :return: The computed determinant
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    for factor_idx, factor in enumerate(matrix[0]):
        sub_matrix = [elem[:factor_idx] + elem[factor_idx + 1:]
                      for elem in matrix[1:]]
        det += (-1)**factor_idx * factor * determinant(sub_matrix)
    return det


def minor(matrix):
    """
    Compute the minor matrix
    :param matrix: The given matrix
    :return: The minor matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(sub_list):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_output = []

    for i in range(len(matrix)):
        inner = []
        for j in range(len(matrix[0])):
            matrix_copy = [x[:] for x in matrix]
            del matrix_copy[i]
            for row in matrix_copy:
                del row[j]
            inner.append(determinant(matrix_copy))
        minor_output.append(inner)

    return minor_output
