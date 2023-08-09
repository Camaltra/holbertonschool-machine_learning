#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Compute the whole forward algorithm given x
    :param rnn_cells: The RNN cells use for the forward pass
    :param X: The data
    :param h_0: The first hidden state
    :return: Both array of states and outputs
    """
    num_of_rnn = len(rnn_cells)
    steps, batch_size, word_dim = X.shape
    _, __, hidden_dim = h_0.shape
    states = np.zeros((steps + 1, num_of_rnn, batch_size, hidden_dim))
    outputs = np.zeros((steps, batch_size, rnn_cells[-1].Wy.shape[1]))
    states[0] = h_0

    for step in range(steps):
        for rnn_idx in range(len(rnn_cells)):
            rnn = rnn_cells[rnn_idx]
            if rnn_idx == 0:
                states[step + 1, rnn_idx], outputs[step] = rnn.forward(
                    states[step, rnn_idx], X[step]
                )
            else:
                states[step + 1, rnn_idx], outputs[step] = rnn.forward(
                    states[step, rnn_idx], states[step + 1, rnn_idx - 1]
                )
    return states, outputs
