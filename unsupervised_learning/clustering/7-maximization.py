#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def maximization(X, g):
    """
    Update the prior, the means and the covariance matrix
    :param X: The datdaset
    :param g: The posterior probs
    :return: The new priors, means and covariance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if g.shape[1] != X.shape[0]:
        return None, None, None
    cluster = np.sum(g, axis=0)
    cluster = np.sum(cluster)
    if int(cluster) != X.shape[0]:
        return None, None, None

    num_of_coords = X.shape[1]
    num_of_clusters = g.shape[0]

    pi = np.sum(g, axis=1) / X.shape[0]

    means = np.zeros(shape=(num_of_clusters, num_of_coords))
    covs = np.zeros(shape=(num_of_clusters, num_of_coords, num_of_coords))
    for cluster in range(num_of_clusters):
        means[cluster] = np.matmul(g[cluster], X) / np.sum(g, axis=1)[cluster]
        norm = X - means[cluster]
        covs[cluster] = (np.matmul(g[cluster] * norm.T, norm) /
                         np.sum(g, axis=1)[cluster])

    return pi, means, covs
