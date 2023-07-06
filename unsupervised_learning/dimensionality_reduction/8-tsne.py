#!/usr/bin/env python3


"""useless comment"""


import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Compute the t-sne dimension reduction
    :param X: Containing the dataset to be transformed by t-SNE
    :param ndims: New dimensional representation of X
    :param idims: Intermediate dimensional representation of X after PCA
    :param perplexity: Is the perplexity
    :param iterations: Is the number of iterations
    :param lr: Is the learning rate
    :return: Y containing the optimized low dimensional transformation of X
    """
    momentum_coeff = 0.8
    n, d = X.shape

    pca_res = pca(X, idims)
    P = P_affinities(pca_res, perplexity=perplexity)
    P *= 4

    Y = []
    y = np.random.randn(n, ndims)
    Y.append(y)
    Y.append(y)

    for i in range(iterations):
        dY, Q = grads(Y[-1], P)
        y = Y[-1] - lr * dY + momentum_coeff * (Y[-1] - Y[-2])
        y = y - np.tile(np.mean(y, 0), (n, 1))
        Y.append(y)

        if (i + 1) % 100 == 0:
            current_cost = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, current_cost))

        if i == 20:
            momentum_coeff = 0.5

        if (i + 1) == 100:
            P /= 4

    return Y[-1]
