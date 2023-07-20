#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def forward(observation, emission, transition, initial):
    """
    Compute the forward algorithm of HMM
    :param observation: The lsit of observation x
    :param emission: The emission matrix p(X | Z)
    :param transition: The transition matrix p(Z+1 | Z)
    :param initial: The inital probs
    :return: The likelihood of the observations given the model
             and the forward path probabilities
    """
    try:
        num_obs = observation.shape[0]
        hidden_state = emission.shape[0]
        F = np.zeros((hidden_state, num_obs))

        F[:, 0] = initial.T * emission[:, observation[0]]

        for i in range(1, num_obs):
            F[:, i] = emission[:, observation[i]] * \
                      np.matmul(F[:, i - 1], transition)

        return np.sum(F[:, num_obs - 1]), F
    except Exception:
        return None, None
