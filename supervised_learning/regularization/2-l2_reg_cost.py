#!/usr/bin/env python3


"""Useless comment"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculate the cost of a neural network unsing L2 regression
    :param cost: The base cost of the neural network
    :return: The full cost (base cost + regularized cose)
    """
    return cost + tf.losses.get_regularization_loss()
