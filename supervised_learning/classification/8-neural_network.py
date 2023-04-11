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
    """Neural network interface"""
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
        self.W1 = np.random.normal(size=(self.nodes, self.nx))
        self.b1 = np.zeros((self.nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, self.nodes))
        self.b2 = 0
        self.A2 = 0
