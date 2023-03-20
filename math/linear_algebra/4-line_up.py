#!/usr/bin/env python3


"""Useless comment"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_arrays(
        first_array: list | None,
        second_array: list | None,
) -> list | None:
    """
    Add two array of the same size
    :param first_array: The fist array
    :param second_array: The second array
    :return: The sum of the arrays, if not the same
             shape or one doesn't exist, None
    """
    if first_array is None or second_array is None:
        return None
    if matrix_shape(first_array) != matrix_shape(second_array):
        return None
    return [first_array_elem + second_array_elem
            for first_array_elem, second_array_elem in
            zip(first_array, second_array)]
