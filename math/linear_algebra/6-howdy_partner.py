#!/usr/bin/env python3


"""Useless comment"""


def cat_arrays(
        first_array: list[int, float] | None,
        second_array: list[int, float] | None,
) -> list[int, float]:
    """
    Concat two array
    :param first_array: The first given array
    :param second_array: The second given array
    :return: Return the concat of the two array or an
             empty list if an array doesn't exist
    """
    if first_array is None or second_array is None:
        return []
    return first_array + second_array
