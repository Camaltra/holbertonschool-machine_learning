#!/usr/bin/env python3

"""useless comment"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate the l2 regularized cost of a NN
    :param cost: The base cost of the NN
    :param lambtha: The param lambda
    :param weights: The weigth of the model (dict)
    :param L: Le size of the NN
    :param m: The number of samples of the data set
    :return: The full cost of the NN (base + regulirazed)
    """
    l2_cost = 0
    for layer_idx in range(L):
        l2_cost += np.sum(weights.get(f"W{layer_idx + 1}")**2)

    return cost + (lambtha / (2 * m)) * l2_cost
