#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Compute the scaled dot product attention
    :param Q: The query Vector
    :param K: The Key vector
    :param V: The value vector
    :param mask: The mask if any
    :return: The attention and the weight
    """
    if mask is not None:
        mask *= -1e9
        weight = tf.math.softmax(
            (tf.matmul(Q, K, transpose_b=True) + mask) / tf.math.sqrt(Q.shape[-1])
        )

    else:
        weight = tf.math.softmax(
            tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(Q.shape[-1])
        )

    attention = tf.matmul(weight, V)

    return attention, weight
