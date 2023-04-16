#!/usr/bin/env python3

import tensorflow as tf

"""Useless comments"""


def create_placeholders(nx, classes):
    """
    Create placeholders tensor
    :param nx: The number of feature columns in our data
    :param classes: The number of classes in our classifier
    :return: The two placeholders
    """
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="x")
    return x, y
