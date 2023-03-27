#!/usr/bin/env python3

"""Useless comment"""


def poly_derivative(polynome):
    """
    Useless comment -- The name of the fuction is enough
    :param polynome: The given polynone
    :return: What the function say
    """
    if not isinstance(polynome, list) or len(polynome) < 1:
        return None
    if len(polynome) == 1:
        return [0]

    derivate = [0] * (len(polynome) - 1)
    for idx in range(len(polynome) - 1, 0, -1):
        derivate[idx - 1] = polynome[idx] * idx
    return derivate
