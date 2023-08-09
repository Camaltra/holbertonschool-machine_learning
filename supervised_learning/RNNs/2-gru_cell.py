#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def sigmoid(x):
    """
    Sigmoid Function
    :param x: The x value
    :return: The computed sigmoid
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax function
    :param x: The x value
    :return: The computed softmax
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class GRUCell:
    """
    Describe a GRU Cell
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Init the class
        :param input_dim: The input x_t dimension
        :param hidden_dim: The hidden h_t dimmension
        :param output_dim: The output y_t dimension
        """
        self.output_dim = output_dim
        self.Wz = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wr = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wh = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wy = np.random.normal(size=(hidden_dim, output_dim))

        self.bz = np.zeros((1, hidden_dim))
        self.br = np.zeros((1, hidden_dim))
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    def forward(self, h_prev, x_t):
        """
        Compute the forward algo for only one step
        :param h_prev: The previous hidden state
        :param x_t: The current input
        :return: The current hidden state, the current output
        """
        input_reset_and_update = np.hstack((h_prev, x_t))
        reset_gate = sigmoid(
            np.matmul(input_reset_and_update, self.Wr) + self.br
        )

        update_gate = sigmoid(
            np.matmul(input_reset_and_update, self.Wz) + self.bz
        )

        h_input = np.hstack((reset_gate * h_prev, x_t))
        h = np.tanh(np.matmul(h_input, self.Wh) + self.bh)

        h_current = update_gate * h + (1 - update_gate) * h_prev

        output = softmax(np.matmul(h_current, self.Wy) + self.by)

        return h_current, output
