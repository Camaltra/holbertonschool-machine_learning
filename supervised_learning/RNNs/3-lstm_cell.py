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


class LSTMCell:
    """
    Describe a LSTM Cell
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Init the class
        :param input_dim: The input x_t dimension
        :param hidden_dim: The hidden h_t dimmension
        :param output_dim: The output y_t dimension
        """
        self.output_dim = output_dim
        self.Wf = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wu = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wc = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wo = np.random.normal(size=(hidden_dim + input_dim, hidden_dim))
        self.Wy = np.random.normal(size=(hidden_dim, output_dim))

        self.bf = np.zeros((1, hidden_dim))
        self.bu = np.zeros((1, hidden_dim))
        self.bc = np.zeros((1, hidden_dim))
        self.bo = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    def forward(self, h_prev, c_prev, x_t):
        """
        Compute the forward algo for only one step
        :param h_prev: The previous hidden state (ShortTerm memory)
        :param c_prev: The previous cell state (LongTerm memory)
        :param x_t: The current input
        :return: The current hidden state, the current output
        """
        input_cell = np.hstack((h_prev, x_t))

        forget_gate = sigmoid(np.matmul(input_cell, self.Wf) + self.bf)
        update_gate = sigmoid(np.matmul(input_cell, self.Wu) + self.bu)
        intermediate_gate = np.tanh(np.matmul(input_cell, self.Wc) + self.bc)
        output_gate = sigmoid(np.matmul(input_cell, self.Wo) + self.bo)

        c_t = (c_prev * forget_gate) + (update_gate * intermediate_gate)
        h_t = np.tanh(c_t) * output_gate
        y = softmax(np.matmul(h_t, self.Wy) + self.by)

        return h_t, c_t, y
