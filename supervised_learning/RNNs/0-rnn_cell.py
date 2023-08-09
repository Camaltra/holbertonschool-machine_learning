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


class RNNCell:
    """
    Describe an RNN Cell
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Init the class
        :param input_dim: The input x_t dimension
        :param hidden_dim: The hidden h_t dimmension
        :param output_dim: The output y_t dimension
        """
        self.output_dim = output_dim
        self.Wh = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wy = np.random.normal(size=(hidden_dim, output_dim))
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    def forward(self, h_prev, x_t):
        """
        Compute the forward algo for only one step
        :param h_prev: The previous hidden state
        :param x_t: The current input
        :return: The current hidden state, the current output
        """
        transformed_x = np.hstack((h_prev, x_t))
        h_t = np.tanh(np.matmul(transformed_x, self.Wh) + self.bh)
        y_output = softmax(np.matmul(h_t, self.Wy) + self.by)

        return h_t, y_output
