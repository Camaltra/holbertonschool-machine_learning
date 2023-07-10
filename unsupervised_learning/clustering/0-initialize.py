#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def initialize(X, k):
    """
    Init the K-mean clusters following a uniform proba law
    :param X: The dataset
    :param k: The number of clusters
    :return: The coordonate of the cluster(s)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    cluster_center_coords = np.random.uniform(X_min, X_max, (k, X.shape[1]))

    return cluster_center_coords
