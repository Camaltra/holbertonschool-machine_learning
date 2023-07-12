#!/usr/bin/env python3


"""useless comment"""


from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Use skleanr Kmean
    :param X: The dataset
    :param k: The number of cluster
    :return: The center of cluster and the labels
    """
    kmeans = KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_