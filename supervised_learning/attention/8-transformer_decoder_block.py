#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder Block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Init the class
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The number of hidden unit
        :param drop_rate: The drop rate for dropout
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation="relu"
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call the layer
        :param x: The data inputed
        :param encoder_output: The encoder output
        :param training: If training or not
        :param look_ahead_mask: First mask if any
        :param padding_mask: Second mask if any
        :return: The output of the decoder
        """
        pass
        attention, _ = self.mha1(x, x, x, look_ahead_mask)
        dropout_attention = self.dropout1(attention)
        attention_x_norm = self.layernorm1(x + dropout_attention)

        attention2, _ = self.mha2(
            attention_x_norm, encoder_output, encoder_output, padding_mask
        )
        dropout_attention2 = self.dropout2(attention2)
        attention2_eoutput_norm = self.layernorm2(
            attention_x_norm + dropout_attention2
        )

        hidden = self.dense_hidden(attention2_eoutput_norm)
        output = self.dense_output(hidden)
        dropout_output = self.dropout2(output)
        output_norm = self.layernorm2(attention2_eoutput_norm + dropout_output)

        return output_norm
