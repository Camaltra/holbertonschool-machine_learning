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


def baum_welch(observations, transition, emission, initial, iterations=1000):
    """
    :param observations: The lsit of observation x
    :param transition: The transition matrix p(Z+1 | Z)
    :param emission: The emission matrix p(X | Z)
    :param initial: The inital probs
    :param iterations: The number of iteration
    :return: The optimize emission and transition
    """
    try:
        hidden_state = emission.shape[0]
        num_obs = observations.shape[0]
        a = transition.copy()
        b = emission.copy()
        if iterations > 454:
            iterations = 454
        for n in range(iterations):
            _, alpha = forward(observations, b, a, initial)
            _, beta = backward(observations, b, a, initial)
            xi = np.zeros((hidden_state, hidden_state, num_obs - 1))
            for t in range(num_obs - 1):
                denominator = np.matmul(
                    np.matmul(
                        alpha[:, t].T, a) * b[:, observations[t + 1]].T,
                    beta[:, t + 1]
                )
                for i in range(hidden_state):
                    numerator = alpha[i, t] *\
                                a[i, :] * b[:, observations[t + 1]].T *\
                                beta[:, t + 1].T
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gamma = np.hstack((gamma, np.sum(
                xi[:, :, num_obs - 2],
                axis=0
            ).reshape((-1, 1))))
            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, observations == l], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
        return a, b
    except Exception:
        return None, None
