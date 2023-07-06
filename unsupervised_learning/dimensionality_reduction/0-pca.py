#!/usr/bin/env python3


"""useless comments"""


import numpy as np


def pca(X, var=0.95):
    """
    Compute the PCA, to get var% of the var explain
    :param X: The X to decompose
    :param var: The var threshold
    :return: THe W matrix
    """
    U, S, Vt = np.linalg.svd(X)

    total_var_explain = 0
    idx = 0

    normal_S = S / np.sum(S)

    for var_explain in normal_S:
        total_var_explain += var_explain
        idx += 1
        if total_var_explain >= var:
            break

    return Vt.T[..., :idx]
