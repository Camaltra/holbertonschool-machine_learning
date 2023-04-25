#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalize a layer unactived input
    Such as gamma * normalized_input + beta
    :param Z: The layer unactived input
    :param gamma: The gamma parameter
    :param beta: The beta parameter
    :param epsilon: A small number to avoid division by 0
    :return: The normalized batch
    """
    mean, std = np.mean(Z, axis=0), np.var(Z, axis=0)
    normilized_z = (Z - mean) / np.sqrt(std + epsilon)
    return gamma * normilized_z + beta
