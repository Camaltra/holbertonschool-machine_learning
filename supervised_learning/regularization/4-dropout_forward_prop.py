#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Compute a forward propagation using the dropout
    regularization
    :param X: The dataset input
    :param weights: A dict of the weights
    :param L: The number of layers
    :param keep_prob: The keep_prob parameter
    :return: The chache: ie all result from each layer
             + all the dropout matrix show all killed
             neurons
    """
    cache = {"A0": X}
    for layer_idx in range(L):
        current_weight = weights.get("W{}".format(layer_idx + 1))
        current_bias = weights.get("b{}".format(layer_idx + 1))
        current_cache_key = "A{}".format(layer_idx + 1)
        current_dropout_key = "D{}".format(layer_idx + 1)
        previous_preds = cache.get("A{}".format(layer_idx))

        Z = np.dot(current_weight, previous_preds) + current_bias

        if layer_idx == L - 1:
            preds = softmax(Z)
        else:
            preds = np.tanh(Z)
            random = np.random.rand(preds.shape[0], preds.shape[1])
            dropout = (random < keep_prob).astype(int)
            preds = np.multiply(preds, dropout) / keep_prob
            cache[current_dropout_key] = dropout

        cache[current_cache_key] = preds

    return cache
