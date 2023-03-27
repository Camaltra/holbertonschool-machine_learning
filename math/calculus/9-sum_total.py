#!/usr/bin/env python3

"""Useless comments"""


def summation_i_squared(n):
    """
    Useless comment -- The name of the fuction is enough
    :param n: The number of iteration
    :return: What the function say
    """
    if n == 1:
        return 1
    return summation_i_squared(n - 1) + n**2
