#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def viterbi(observation, emission, transition, initial):
    """
    Compute the viterbi algorithm
    :param observation: The lsit of observation x
    :param emission: The emission matrix p(X | Z)
    :param transition: The transition matrix p(Z+1 | Z)
    :param initial: The inital probs
    :return: The path of most likely sequence of hidden states
             Probability of obtaining the path sequence
    """
    try:
        num_obs = observation.shape[0]
        hidden_state = emission.shape[0]

        T1 = np.empty((hidden_state, num_obs), 'd')
        T2 = np.empty((hidden_state, num_obs), 'B')

        T1[:, 0] = initial.T * emission[:, observation[0]]
        T2[:, 0] = 0

        for i in range(1, num_obs):
            T1[:, i] = np.max(
                T1[:, i - 1] * emission[np.newaxis, :, observation[i]].T
                * transition.T, 1)
            T2[:, i] = np.argmax(T1[:, i - 1] * transition.T, 1)

        x = np.empty(num_obs, 'B')
        x[-1] = np.argmax(T1[:, num_obs - 1])
        for i in reversed(range(1, num_obs)):
            x[i - 1] = T2[x[i], i]

        return list(x), np.max(T1[:, num_obs - 1])
    except Exception:
        return None, None
