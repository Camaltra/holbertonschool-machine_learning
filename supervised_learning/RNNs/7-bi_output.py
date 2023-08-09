#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def softmax(x):
    """
    Softmax function
    :param x: The x value
    :return: The computed softmax
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class BidirectionalCell:
    """
    Describe an Bidirectionnal RNN Cell
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Init the class
        :param input_dim: The input x_t dimension
        :param hidden_dim: The hidden h_t dimmension
        :param output_dim: The output y_t dimension
        """
        self.output_dim = output_dim
        self.Whf = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Whb = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wy = np.random.normal(size=(2 * hidden_dim, output_dim))
        self.bhf = np.zeros(shape=(1, hidden_dim))
        self.bhb = np.zeros(shape=(1, hidden_dim))
        self.by = np.zeros(shape=(1, output_dim))

    def forward(self, h_prev, x_t):
        """
        Compute the forward algo for only one step
        :param h_prev: The previous hidden state
        :param x_t: The current input
        :return: The current hidden state, the current output
        """
        transformed_x = np.hstack((h_prev, x_t))
        h_t = np.tanh(np.matmul(transformed_x, self.Whf) + self.bhf)
        return h_t

    def backward(self, h_next, x_t):
        """
        Compute the backward algo for only one step
        :param h_next: The last hidden states (The first one in reverse)
        :param x_t: The current input
        :return: The current state, the current output
        """
        transformed_x = np.hstack((h_next, x_t))
        h_t = np.tanh(np.matmul(transformed_x, self.Whb) + self.bhb)
        return h_t

    def output(self, H):
        """
        Compute the output
        :param H: The concatenate array of both forward and backward states
        :return: The whole output (For each timestamp)
        """
        timestamp, batch_size, __ = H.shape

        outputs = np.zeros(shape=(timestamp, batch_size, self.output_dim))
        for t in range(timestamp):
            outputs[t] = softmax(np.matmul(H[t], self.Wy) + self.by)

        return outputs
