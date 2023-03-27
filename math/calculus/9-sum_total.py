#!/usr/bin/env python3

"""Useless comments"""


def summation_i_squared(n):
    """
    Useless comment -- The name of the fuction is enough
    :param n: The number of iteration
    :return: What the function say
    """
    if not isinstance(n, int) or n < 1:
        return None

    return int(n * (n + 1) * (2 * n + 1) / 6)
