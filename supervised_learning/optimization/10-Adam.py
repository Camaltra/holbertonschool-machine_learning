#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Define the training operator using Adam optimizer
    :param loss: The loss function
    :param alpha: The learning rate
    :param beta1: The momentum weight
    :param beta2: The RMS weigth
    :param epsilon: A small number to not divide by 0
    :return: The train op
    """
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)
