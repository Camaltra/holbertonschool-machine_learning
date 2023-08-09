#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Compute the whole forward algorithm given x
    :param rnn_cell: The RNN cell use for the forward pass
    :param X: The data
    :param h_0: The first hidden state
    :return: Both array of states and outputs
    """
    steps, batch_size, word_dim = X.shape
    _, hidden_dim = h_0.shape

    states = np.zeros((steps + 1, batch_size, hidden_dim))
    states[0] = h_0
    outputs = np.zeros((steps, batch_size, rnn_cell.Wy.shape[1]))

    for step in range(steps):
        current_state, current_output = rnn_cell.forward(
            states[step], X[step, :, :]
        )
        states[step + 1] = current_state
        outputs[step] = current_output

    return states, outputs
