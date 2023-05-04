#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Compute the dropout gradient descent
    :param X: The input datasets
    :param weights: A dict of weights
    :param L: The size of the NN
    :param keep_prob: The keep_prob parameter
    :return: Nothing
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
        if layer_idx < L:
            current_droupout_key = "D{}".format(layer_idx)
            current_dropout_matrix = cache.get(current_droupout_key)
            dZ *= current_dropout_matrix / keep_prob

        dW = np.dot(dZ, previous_preds.T) / num_of_sample
        db = np.sum(dZ, axis=1, keepdims=True) / num_of_sample

        weights[current_weight_key] = current_weight - alpha * dW
        weights[current_bias_key] = current_bias - alpha * db
        if layer_idx > 1:
            dZ = np.dot(
                current_weight.T, dZ
            ) * (1 - (previous_preds * previous_preds))
