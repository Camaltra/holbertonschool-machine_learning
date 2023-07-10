#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def compute_centroid_distance(X, centroids_coords):
    """
    Compute the distance between a point and the K centroids
    :param X: The dataset
    :param centroids_coords: The centroids coords
    :return: The distrance betweens points and centroids
    """
    return np.sqrt(np.sum((X - centroids_coords[:, np.newaxis]) ** 2, axis=2))


def variance(X, C):
    """
    Compute the variance for the Kmean algos
    :param X: The dataset
    :param C: The cluster center points
    :return: The computed variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if (not isinstance(C, np.ndarray) or len(C.shape) != 2 or
            C.shape[1] != X.shape[1]):
        return None
    points_centroids_distance = compute_centroid_distance(X, C)
    cluster_groups = np.min(points_centroids_distance, axis=0)

    return np.sum(np.square(cluster_groups))
