#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

positional_encoding = __import__("4-positional_encoding").positional_encoding
DecoderBlock = __import__("8-transformer_decoder_block").DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder Block"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_len,
            drop_rate=0.1
    ):
        """
        Init the class
        :param N: The number of decoder
        :param dm: The model detph
        :param h: The number of head
        :param hidden: The hidden units
        :param target_vocab: The target vocab list (For the embedding)
        :param max_seq_len: The max sequence length
        :param drop_rate: The drop rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call the layer
        :param x: The data inputed
        :param encoder_output: The encoder output
        :param training: If training or not
        :param look_ahead_mask: The first mask if any
        :param padding_mask: The second mask if any
        :return:
        """
        x = self.embedding(x) + self.positional_encoding[: x.shape[1]]
        x = self.dropout(x, training=training)
        for dblock in self.blocks:
            x = dblock(
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask
            )
        return x
