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


def cofactor(matrix):
    """
    Compute the cofactor of the matrix
    :param matrix: The given matrix
    :return: The cofactor matrix
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

    matrix = minor(matrix)
    for i in range(len(matrix)):
        sign = (-1)**(i)
        for j in range(len(matrix[0])):
            matrix[i][j] *= sign
            sign *= -1

    return matrix


def adjugate(matrix):
    """

    :param matrix:
    :return:
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

    matrix = cofactor(matrix)

    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]


def inverse(matrix):
    """
    Get the inverse of a matrix
    :param matrix: The given Matrix
    :return: The inverse of the matrix
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
    det = determinant(matrix)
    if det == 0:
        return None
    adjugate_matrix = adjugate(matrix)
    return [[adjugate_matrix[i][j]/det for j in range(len(adjugate_matrix[0]))]
            for i in range(len(adjugate_matrix))]
