#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update variable using the adam descent gradient technique
    :param alpha: The learning rate
    :param beta1: The momuntum weight
    :param beta2: The RMS weight
    :param epsilon: A small number to not divde by 0
    :param var: Contain the variable to be updated
    :param grad: Contain the gradient of var
    :param v: The previous first moment of var
    :param s: The previous second moment of var
    :param t: The time step used for bias correction
    :return: The updated variable, the new first moment,
             and the new second moment, respectively
    """
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad * grad
    vdw_corrected = vdw / (1 - beta1 ** t)
    sdw_corrected = sdw / (1 - beta2 ** t)

    corrected_ratio = vdw_corrected / (np.sqrt(sdw_corrected) + epsilon)

    updated_w = var - alpha * corrected_ratio

    return updated_w, vdw, sdw
