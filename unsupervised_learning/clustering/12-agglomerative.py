#!/usr/bin/env python3


"""useless comment"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Compute a clustering using the agglomerative algos
    :param X: The dataset
    :param dist: The maximum cophenetic distance for all clusters
    :return: The class for each data point
    """

    hierarchy = scipy.cluster.hierarchy
    links = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(links, t=dist, criterion='distance')

    plt.figure()
    hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    return clss