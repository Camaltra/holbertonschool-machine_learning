#!/usr/bin/env python3

"""Useless comment"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta, epsilon):
    """
    Define the training operator for the RMS
    :param loss: The loss function tensor
    :param alpha: The learning rate
    :param beta: The RMSProp weight
    :param epsilon: A small number to avoid division by zero
    :return: The training momentum function
    """
    return tf.train.RMSPropOptimizer(alpha,
                                     decay=beta,
                                     epsilon=epsilon).minimize(loss)
