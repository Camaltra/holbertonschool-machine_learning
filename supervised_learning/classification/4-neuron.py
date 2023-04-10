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
    def A(self) -> int | np.ndarray:
        """
        Get the private attribe A
        :return: Private attribe A
        """
        return self.__A

    def forward_prop(self, X: np.ndarray):
        """
        Set the forward propagation of the given neuron
        Wierd thing here about X and W, they sould be inverted,
        cause of the fact that they consider the X matrix as:
            - Columns as samples
            - Row as features
        :param X: The data set
        :return: The result of the neuron fuction sigma(w0 + w1x1 + ... + wnxn)
        """
        pre_processed_data = np.dot(self.W, X) + self.b
        self.__A = np.vectorize(
            lambda x: 1 / (1 + np.exp(-x))
        )(pre_processed_data)
        return self.A

    def cost(self, Y: np.ndarray, A: np.ndarray) -> float:
        """
        Calculate the cost function form the neuron, we consider the neuron as
        a logistic regression model as it use the logistic function to
        make prediction
        :param Y: The thuth from the data
        :param A: The predict data
        Again the data is shaped like sample are on the columns
        :return: The cost of the function
        """
        num_of_sample = A.shape[1]
        log_loss_res = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return -num_of_sample ** -1 * np.sum(log_loss_res)

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        """
        Evaluate the prediction made by the neuron
        :param X: The data set
        :param Y: The thuth labels
        :return: The predictions and the cost
        """
        self.forward_prop(X)
        preds = np.where(self.A < 0.5, 0, 1)
        return preds, self.cost(Y, self.A)
