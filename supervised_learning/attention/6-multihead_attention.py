#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

sdp_attention = __import__("5-sdp_attention").sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi Head Attention"""

    def __init__(self, dm, h):
        """
        Init the class
        :param dm: The model depth
        :param h: Number of head
        """
        super().__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def _split_heads(self, head, batch_size):
        """
        Split the given layer to multiple heads
        :param head: The linear layer
        :param batch_size: The batch_size
        :return: Tensor of all the heads
        """
        x = tf.reshape(head, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Call the layer
        :param Q: The query
        :param K: The keys
        :param V: The values
        :param mask: The mask if any
        :return: The output and scaled weights
        """
        batch_size = Q.shape[0]

        q = self._split_heads(self.Wq(Q), batch_size)
        k = self._split_heads(self.Wk(K), batch_size)
        v = self._split_heads(self.Wv(V), batch_size)

        scaled_attention, scaled_weight = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        scaled_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm)
        )

        output = self.linear(scaled_attention)

        return output, scaled_weight
