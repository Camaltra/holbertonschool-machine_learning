#!/usr/bin/env python3


"""useless comments"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def _process_binary_search(D, betas, H, tol, i):
    """
    Process binary search iterativly
    :param D: The pairwise distance
    :param betas: The betas from the gaussian distrib
    :param H: The Shannon entropy given perpexity
    :param tol: maximum tolerance allowed (inclusive)
                for the difference in Shannon entropy
    :param i: The idex of the current point
    :return:
    """
    Di = D[i].copy()
    Di = np.delete(Di, i, axis=0)
    Hi, Pi = HP(Di, betas[i])

    H_diff = H - Hi

    upper = None
    lower = None

    while np.abs(H_diff) > tol:
        if H_diff > 0:
            upper = betas[i, 0]
            if lower is None:
                betas[i] = betas[i] / 2
            else:
                betas[i] = (betas[i] + lower) / 2

        else:
            lower = betas[i, 0]
            if upper is None:
                betas[i] = betas[i] * 2
            else:
                betas[i] = (betas[i] + upper) / 2

        Hi, Pi = HP(Di, betas[i])
        H_diff = H - Hi
    return Pi


def P_affinities(X, tol=1e-5, perplexity=30.0):
    D, P, betas, H = P_init(X, perplexity)

    n, _ = D.shape

    for i in range(n):
        Pi = _process_binary_search(D, betas, H, tol, i)
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi

    return (P.T + P) / (2*n)
