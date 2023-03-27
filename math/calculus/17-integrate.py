#!/usr/bin/env python3

"""Useless comment"""


def poly_integral(polynome, primitive_const=0):
    """
    Useless comment -- The name of the fuction is enough
    :param polynome: The given polynone
    :param primitive_const: The integration constant
    :return: What the function say
    """
    if not isinstance(polynome, list) or len(polynome) < 1:
        return None
    if not isinstance(primitive_const, int) and \
            not isinstance(primitive_const, float):
        return None

    primitive = [0] * (len(polynome) + 1)
    for idx in range(len(polynome) - 1, -1, -1):
        primitive[idx + 1] = polynome[idx] / (idx + 1)
    primitive[0] = primitive_const
    return primitive
