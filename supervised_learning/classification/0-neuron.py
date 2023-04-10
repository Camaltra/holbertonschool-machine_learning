#!/usr/bin/env python3


"""Useless comment"""

import numpy as np


def _check_nx(nx: int) -> None:
    """
    Check is the nuber of inputed features are a int and only positif
    :param nx: The number of inputed features
    :return: Nothing but raise exception if not good value
    """
    if not isinstance(nx, int):
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")


class Neuron:
    """
    Class basic neuron
    """
    def __init__(self, nx: int) -> None:
        """
        Init a basic neuron
        :param nx: The nuber of inputed features
        """
        _check_nx(nx)
        self.nx = nx
        self.W = np.random.normal(size=(1, self.nx))
        self.b = 0
        self.A = 0
