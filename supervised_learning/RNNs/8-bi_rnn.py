#!/usr/bin/env python3


"""useless comment"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Compute the whole forward algorithm given x
    :param bi_cell: The Bi-RNN cell use for the forward pass
    :param X: The data
    :param h_0: The first hidden state
    :param h_t: The last hidden state
    :return: Both array of states and outputs
    """
    steps, batch_size, word_dim = X.shape
    _, hidden_dim = h_0.shape

    forward_states = np.zeros((steps + 1, batch_size, hidden_dim))
    forward_states[0] = h_0
    backward_states = np.zeros((steps + 1, batch_size, hidden_dim))
    backward_states[-1] = h_t
    # outputs = np.zeros((steps, batch_size, bi_cell.output_dim))

    for step in range(steps):
        current_state = bi_cell.forward(forward_states[step], X[step, :, :])
        forward_states[step + 1] = current_state

    for step in reversed(range(steps)):
        current_state = bi_cell.backward(
            backward_states[step + 1], X[step, :, :]
        )
        backward_states[step] = current_state

    states = np.concatenate((forward_states[1:], backward_states[:-1]), axis=2)
    outputs = bi_cell.output(states)

    return states, outputs
