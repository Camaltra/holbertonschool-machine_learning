#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe
    side effects
    :param x: is the number of patients that develop
              severe side effects
    :param n: is the total number of patients observed
    :param P: is a 1D numpy.ndarray containing the various
              hypothetical probabilities of developing
              severe side effects
    :return: The posterior
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coeff = (np.math.factorial(n) /
                      (np.math.factorial(x) * np.math.factorial(n - x)))

    return binomial_coeff * P ** x * (1 - P) ** (n - x)
