#!/usr/bin/env python3

"""useless comments"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Create a training operation
    :param loss: The loss of the network’s prediction
    :param alpha: The learning rate
    :return: Operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
