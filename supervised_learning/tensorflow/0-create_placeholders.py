#!/usr/bin/env python3

"""Useless comments"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Create placeholders tensor
    :param nx: The number of feature columns in our data
    :param classes: The number of classes in our classifier
    :return: The two placeholders
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y
