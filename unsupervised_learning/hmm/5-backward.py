#!/usr/bin/env python3


"""Useless comment"""


import numpy as np


def backward(observation, emission, transition, initial):
    """
    Compute the backward probabilities
    :param observation: The lsit of observation x
    :param emission: The emission matrix p(X | Z)
    :param transition: The transition matrix p(Z+1 | Z)
    :param initial: The inital probs
    :return: The likelihood of the observations given the model
             The backward path probabilities
    """
    try:
        num_obs = observation.shape[0]
        hidden_state = emission.shape[0]
        B = np.zeros((hidden_state, num_obs))

        B[:, num_obs - 1] = np.ones(shape=(1, hidden_state))

        for i in range(num_obs - 2, -1, -1):
            B[:, i] = np.matmul(
                emission[:, observation[i + 1]] * transition,
                B[:, i + 1]
            )

        P = np.sum(initial.T * emission[:, observation[0]] * B[:, 0])
        return P, B
    except Exception:
        return None, None
