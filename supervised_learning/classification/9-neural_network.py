#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def _check_nx(nx: int) -> None:
    """
    Check is the number of inputed features are an int and only positif
    :param nx: The number of inputed features
    :return: Nothing but raise exception if not good value
    """
    if not isinstance(nx, int):
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")


def _check_nodes(nodes: int) -> None:
    """
    Check is the number of required nodes are an int and only positif
    :param nodes: The number of inputed features
    :return: Nothing but raise exception if not good value
    """
    if not isinstance(nodes, int):
        raise TypeError("nodes must be an integer")
    if nodes < 1:
        raise ValueError("nodes must be a positive integer")


class NeuralNetwork:
    def __init__(self, nx: int, nodes: int):
        """
        Init the class
        :param nx: The nu,ber of feature of the data set
        :param nodes: The numbers of node in the hidden layer
        """
        _check_nx(nx)
        _check_nodes(nodes)
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.normal(size=(self.nodes, self.nx))
        self.__b1 = 0
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, self.nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Get the weight for the hidden layer
        :return: The weight of the hidden layer
        """
        return self.__W1

    @property
    def A1(self):
        """
        Get the output predicts for the hidden layer
        :return: The output predicts of the hidden layer
        """
        return self.__A1

    @property
    def b1(self):
        """
        Get the bias for the hidden layer
        :return: The bias of the hidden layer
        """
        return self.__b1

    @property
    def W2(self):
        """
        Get the weight for the output layer
        :return: The weight of the output layer
        """
        return self.__W2

    @property
    def A2(self):
        """
        Get the output predicts for the output layer
        :return: The output predicts of the output layer
        """
        return self.__A2

    @property
    def b2(self):
        """
        Get the bias for the output layer
        :return: The bias of the output layer
        """
        return self.__b2
