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
        self.__W = np.random.normal(size=(1, self.nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self) -> np.ndarray:
        """
        Get the private attribe W
        :return: Private attribe W
        """
        return self.__W

    @property
    def b(self) -> int:
        """
        Get the private attribe b
        :return: Private attribe b
        """
        return self.__b

    @property
    def A(self) -> int:
        """
        Get the private attribe A
        :return: Private attribe A
        """
        return self.__A
