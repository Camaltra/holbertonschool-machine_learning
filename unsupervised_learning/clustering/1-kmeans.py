#!/usr/bin/env python3


"""useless comment"""


import numpy as np
initialize = __import__('0-initialize').initialize


def compute_centroid_distance(X, centroids_coords):
    """
    Compute the distance between a point and the K centroids
    :param X: The dataset
    :param centroids_coords: The centroids coords
    :return: The distrance betweens points and centroids
    """
    return np.sqrt(np.sum((X - centroids_coords[:, np.newaxis]) ** 2, axis=2))


def kmeans(X, k, iterations=1000):
    """
    Run the full Kmean algorithms
    :param X: The dataset
    :param k: The number of clusters
    :param iterations: The number of iterations
    :return: The computed cluster centers | The class for each points
    """
    if not isinstance(X, np.ndarray) or k <= 0 or iterations <= 0:
        return None, None

    centroids = initialize(X, k)

    for i in range(iterations):
        centroids_copy = centroids.copy()
        points_centroids_distance = compute_centroid_distance(X, centroids)
        clss = np.argmin(points_centroids_distance, axis=0)

        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[clss == j], axis=0)

        points_centroids_distance = compute_centroid_distance(X, centroids)
        clss = np.argmin(points_centroids_distance, axis=0)
        if np.all(centroids_copy == centroids):
            break

    return centroids, clss
