#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Compute the gradient descent for the deep neural network with
    regularization
    :param Y: The thruth label
    :param weights: The dict of the weight of the NN
    :param cache: The cache contain the preds for each layers
    :param alpha: Teh learning rate
    :param lambtha: The lambda parameter for the regularization
    :param L: The number of layer
    """
    dZ = cache.get("A{}".format(L)) - Y
    num_of_sample = Y.shape[1]

    for layer_idx in reversed(range(1, L + 1)):
        current_weight_key = "W{}".format(layer_idx)
        current_weight = weights.get(current_weight_key)
        current_bias_key = "b{}".format(layer_idx)
        current_bias = weights.get(current_bias_key)
        previous_preds_key = "A{}".format(layer_idx - 1)
        previous_preds = cache.get(previous_preds_key)

        dW = np.dot(dZ, previous_preds.T) / num_of_sample
        db = np.sum(dZ, axis=1, keepdims=True) / num_of_sample

        l2_reg_param = (lambtha / num_of_sample) * current_weight

        weights[current_weight_key] = current_weight - alpha * (dW + l2_reg_param)
        weights[current_bias_key] = current_bias - alpha * db
        if layer_idx > 1:
            dZ = np.dot(
                current_weight.T, dZ
            ) * (1 - (previous_preds * previous_preds))
