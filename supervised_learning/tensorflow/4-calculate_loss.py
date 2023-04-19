#!/usr/bin/env python3

"""useless comments"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    As the function say, calculate the loss
    :param y: The thuth label
    :param y_pred: The predicted label
    :return: The tensof of predicted value
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
