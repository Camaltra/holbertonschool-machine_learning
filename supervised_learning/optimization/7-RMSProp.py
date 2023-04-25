#!/usr/bin/env python3

"""Useless comment"""

import numpy as np

def update_variables_RMSProp(alpha, beta, epsilon, var, grad, s):
    """

    :param alpha: The learning rate
    :param beta: The RMSProp weight
    :param epsilon: A small number to not devide by 0
    :param var: Contain the variable to be updated
    :param grad: Contain the gradient of var
    :param s: The previous second moment of var
    :return: The updated variable and the new moment, respectively
    """
    sdw = beta * s + (1 - beta) * (grad * grad)
    return var - (grad / (np.sqrt(sdw) + epsilon)) * alpha, sdw
