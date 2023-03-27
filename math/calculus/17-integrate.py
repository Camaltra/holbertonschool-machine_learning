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
    if len(polynome) == 1 and polynome[0] == 0:
        return [C]

    primitive = [0] * (len(polynome) + 1)
    for idx in range(len(polynome) - 1, -1, -1):
        idx_result = polynome[idx] / (idx + 1)
        primitive[idx + 1] = int(idx_result) if idx_result % 1 == 0 else idx_result
    primitive[0] = C
    return primitive
