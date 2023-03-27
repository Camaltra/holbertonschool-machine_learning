#!/usr/bin/env python3

"""Useless comment"""


def poly_integral(polynome, C=0):
    """
    Useless comment -- The name of the fuction is enough
    :param polynome: The given polynone
    :param C: The integration constant
    :return: What the function say
    """
    if not isinstance(polynome, list) or len(polynome) < 1:
        return None
    if not isinstance(C, int) and \
            not isinstance(C, float):
        return None

    primitive = [0] * (len(polynome) + 1)
    for idx in range(len(polynome) - 1, -1, -1):
        primitive[idx + 1] = polynome[idx] / (idx + 1)
    primitive[0] = C
    return primitive
