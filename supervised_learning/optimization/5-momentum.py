#!/usr/bin/env python3

"""Useless comment"""


def update_variables_momentum(alpha, beta, var, grad, v):
    """
    Using the momentum to calculate the gradient descente
    :param alpha: The learning rate
    :param beta: The momentum weight
    :param var: Contain the variable to be updated
    :param grad: Contain the gradient of var
    :param v: The previous irst moment of var
    :return: The updated variable and the new moment, respectively
    """
    dw = beta * v + (1 - beta) * grad
    return var - dw * alpha, dw
