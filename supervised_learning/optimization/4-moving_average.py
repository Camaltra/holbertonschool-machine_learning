#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def moving_average(data, beta):
    """
    Calculate the mouving average
    :param data: The data array
    :param beta: The beta parameter (weigth)
    :return: The array of the mouving average
    """
    result = []
    theta = 1 - beta
    total = 0

    for index, point in enumerate(data, start=1):
        bias_correction = 1 - beta ** index
        total = total * beta + point * theta
        result.append(total / bias_correction)
    return result
