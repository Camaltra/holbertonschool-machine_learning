#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """

    :param alpha:
    :param decay_rate:
    :param global_step:
    :param decay_step:
    :return:
    """
    return tf.compat.v1.train.inverse_time_decay(alpha, global_step, decay_step,
                                                 decay_rate, staircase=True)
