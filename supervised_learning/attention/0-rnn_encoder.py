#!/usr/bin/env python3


"""useless comment"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Init the class
        :param vocab: The vocab size (For the embedding)
        :param embedding: The embbeding dim
        :param units: The num of GRU unit
        :param batch: The batch size
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def initialize_hidden_state(self):
        """
        Init hte first hidden state (tensor)
        :return: The init tensor
        """
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """
        Call the layer
        :param x: The data inputed
        :param initial: The initial state
        :return: The output and all hidden state
        """
        x_embedded = self.embedding(x)
        output, hidden = self.gru(x_embedded, initial_state=initial)
        return output, hidden
