#!/usr/bin/env python3

"""useless comments"""


import tensorflow as tf


def calculate_accuracy(y, y_preds):
    """
    As the function say, calculate the accuracy
    :param y: The thuth label
    :param y_pred: The predicted label
    :return: The tensof of predicted value
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_preds, 1))
    return tf.reduce_mean(tf.cast(accuracy, tf.float32))
