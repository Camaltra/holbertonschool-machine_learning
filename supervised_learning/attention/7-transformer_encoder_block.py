#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

MultiHeadAttention = __import__("6-multihead_attention").MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder Block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Init the class
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden unit
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden,
            activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Call the layer
        :param x: The data inputed
        :param training: If training or not
        :param mask: The mask if any
        :return: The output of the encoder
        """
        attention, _ = self.mha(x, x, x, mask)
        dropout = self.dropout1(attention, training=training)
        attention_x_norm = self.layernorm1(x + dropout)

        hidden = self.dense_hidden(attention_x_norm)
        output = self.dense_output(hidden)

        output = self.dropout2(output)
        outpout = self.layernorm2(attention_x_norm + output)

        return outpout
