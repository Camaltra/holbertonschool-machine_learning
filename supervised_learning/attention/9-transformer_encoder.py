#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Encoder"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_len,
            drop_rate=0.1
    ):
        """
        Init the class
        :param N: The number of encoder
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The hidden unit
        :param input_vocab: The vocab for the embedding part
        :param max_seq_len: The max seq length
        :param drop_rate: The drop rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call the layer
        :param x: The inputed data
        :param training: If training or not
        :param mask: The mask if any
        :return: The output of the decoder
        """
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[: x.shape[1]]
        x = self.dropout(x, training=training)
        for eblock in self.blocks:
            x = eblock(x, training, mask)

        return x
