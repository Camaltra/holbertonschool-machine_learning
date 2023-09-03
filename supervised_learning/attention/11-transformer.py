#!/usr/bin/env python3

"""useless comment"""


import tensorflow as tf

Encoder = __import__("9-transformer_encoder").Encoder
Decoder = __import__("10-transformer_decoder").Decoder


class Transformer(tf.keras.layers.Layer):
    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1,
    ):
        """
        Init the class
        :param N: The number of encoder and decoder
        :param dm: The model depth
        :param h: The number of heads
        :param hidden: The hidden units
        :param input_vocab: The input vocab
        :param target_vocab: The output vocab
        :param max_seq_input: The max_seq_input
        :param max_seq_target: The max_seq_target
        :param drop_rate: The drop rate
        """
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.encoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        """
        Call the layer
        :param inputs: The data input
        :param target: The data target
        :param training: If training or not
        :param encoder_mask: The encoder mask
        :param look_ahead_mask: The first decoder mask
        :param decoder_mask: The second decoder mask
        :return:
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask
        )
        output = self.linear(dec_output)

        return output
