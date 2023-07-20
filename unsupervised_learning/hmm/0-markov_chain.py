#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Compute the markov chain for t iterations
    :param P: The transition matrix
    :param s: The probability of starting in each state
    :param t: The number of iteration
    :return:
    """
    try:
        for _ in range(t):
            s = np.matmul(s, P)
        return s
    except Exception:
        return None
