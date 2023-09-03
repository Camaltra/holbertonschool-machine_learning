#!/usr/bin/env python3


"""useless comment"""


import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Init the class
        :param vocab: The vocab size (For the embedding)
        :param embedding: The embbeding dim
        :param units: The num of GRU unit
        :param batch: The batch size
        """
        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call the laver
        :param x: The data inputed
        :param s_prev: The previous state
        :param hidden_states: The encoder hidden states
        :return: The output of the layer, and the hidden state
        """
        context_vector, attention_weights = self.attention(s_prev,
                                                           hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)
        return x, state
