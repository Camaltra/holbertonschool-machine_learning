#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf

def create_momentum_op(loss, alpha, beta):
    """
    Define the training operator for the momentum
    :param loss: The loss function tensor
    :param alpha: The learning rate
    :param beta: The momentum
    :return: The training momentum function
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta).minimize(loss)
