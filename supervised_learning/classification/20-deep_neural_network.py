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


def _check_layers(layers: int) -> None:
    """
    Check is the number of required nodes are an int and only positif
    :param layers: The list of number of node in each layers
    :return: Nothing but raise exception if not good value
    """
    if not isinstance(layers, list) or len(layers) == 0:
        raise TypeError("layers must be a list of positive integers")
    """ Can't use the for loop here omg this is so stupid """
    """ if not all(nodes > 0 for nodes in layers):
          raise ValueError("layers must be a list of positive integers")"""


class DeepNeuralNetwork:
    """Deep neural network interface"""
    def __init__(self, nx, layers):
        """
        Init the deep neural network and use a dict to store data
        such as the weight, bias and cache data
        :param nx: The nuber of features
        :param layers: An array taht contain the number of
                       nodes for each layers
        """
        _check_nx(nx)
        _check_layers(layers)
        self.__nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer_idx in range(self.L):
            # OMG I WANNA DIE WHY WE CAN'T USE ONLY ONE FOR ??
            if layers[layer_idx] <= 0:
                raise TypeError("layers must be a list of positive integers")
            current_weight_key = "W{}".format(layer_idx + 1)
            current_bias_key = "b{}".format(layer_idx + 1)
            if layer_idx == 0:
                self.weights[current_weight_key] = np.random.randn(
                    layers[layer_idx], self.nx
                ) * np.sqrt(2. / self.nx)
            else:
                self.weights[current_weight_key] = np.random.randn(
                    layers[layer_idx], layers[layer_idx - 1]
                ) * np.sqrt(2. / layers[layer_idx - 1])

            self.weights[current_bias_key] = np.zeros((layers[layer_idx], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def nx(self):
        return self.__nx

    def forward_prop(self, X):
        """
        Compute the forward propagation of the deep neural network
        :param X: The data set
        :return: The result of the forward propagation and the cache
                 for the result of each layers
        """
        self.__cache["A0"] = X
        for layer_idx in range(self.L):
            input_key = "A{}".format(layer_idx)
            weight_key = "W{}".format(layer_idx + 1)
            bias_key = "b{}".format(layer_idx + 1)
            z = np.dot(
                self.weights.get(weight_key),
                self.cache.get(input_key)
            ) + self.weights.get(bias_key)
            A = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(z)
            self.__cache["A{}".format(layer_idx + 1)] = A

        return A, self.cache

    def cost(self, Y, A):
        """
        Compute the cost function for the logistic function
        :param Y: The thruth labels
        :param A: The predictions
        :return: The result of the cost function
        """
        num_of_sample = Y.shape[1]
        return - np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / num_of_sample

    def evaluate(self, X, Y):
        """
        Evaluate the model
        :param X: The data set
        :param Y: The thruth label
        :return: The array of the predictions, and the cost function res
        """
        preds, *_ = self.forward_prop(X)
        return np.where(preds < 0.5, 0, 1), self.cost(Y, preds)
