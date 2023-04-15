#!/usr/bin/env python3

"""Useless comment"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def _check_activation(activation: str):
    """
    Check the activaiton parameter
    :param activation: The activation parameter
    :return: Nothing
    """
    if activation != "sig" and activation != "tanh":
        raise ValueError("activation must be 'sig' or 'tanh'")


def _check_step(step: int, iterations: int) -> None:
    """
    Check the step parameter
    :param step: The step parameter
    :param iterations: The number of overall iteration
    :return: Nothing
    """
    if not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step <= 0 or step > iterations:
        raise ValueError("step must be positive and <= iterations")


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


def _check_iterations(interations: int) -> None:
    """
    Chec the iteration variable
    :param interations: The number of gradient descente iteration
    :return: Nothing
    """
    if not isinstance(interations, int):
        raise TypeError("iterations must be an integer")
    if interations <= 0:
        raise ValueError("iterations must be a positive integer")


def _check_alpha(alpha: float) -> None:
    """
    Check the learning rate parameter
    :param alpha: The learning rate parameter
    :return: Nothing
    """
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")
    if alpha <= 0:
        raise ValueError("alpha must be positive")


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
    def __init__(self, nx, layers, activation='sig'):
        """
        Init the deep neural network and use a dict to store data
        such as the weight, bias and cache data
        :param nx: The nuber of features
        :param layers: An array taht contain the number of
                       nodes for each layers
        """
        _check_nx(nx)
        _check_layers(layers)
        _check_activation(activation)
        self.__nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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
    def activation(self):
        return self.__activation

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
            z = np.matmul(
                self.weights.get(weight_key),
                self.cache.get(input_key)
            ) + self.weights.get(bias_key)
            if layer_idx == self.L - 1:
                A = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
            else:
                if self.activation == "sig":
                    A = 1 / (1 + np.exp(-z))
                else:
                    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            # The checker doesn't liek this line
            # A = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(z)
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
        return - np.sum(Y * np.log(A)) / num_of_sample

    def evaluate(self, X, Y):
        """
        Evaluate the model
        :param X: The data set
        :param Y: The thruth label
        :return: The array of the predictions, and the cost function res
        """
        preds, *_ = self.forward_prop(X)
        cost = self.cost(Y, preds)
        return np.where(preds == np.amax(preds, axis=0), 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Compute the gradient descent for the deep neural network
        :param Y: The thruth label
        :param cache: The cache contain the preds for each layers
        :param alpha: Teh learning rate
        :return: Nothing
        """
        dZ = self.cache.get("A{}".format(self.L)) - Y
        num_of_sample = Y.shape[1]

        for layer_idx in reversed(range(1, self.L + 1)):
            current_weight_key = "W{}".format(layer_idx)
            current_weight = self.weights.get(current_weight_key)
            current_bias_key = "b{}".format(layer_idx)
            current_bias = self.weights.get(current_bias_key)
            previous_preds_key = "A{}".format(layer_idx - 1)
            previous_preds = self.cache.get(previous_preds_key)

            dW = np.matmul(dZ, previous_preds.T) / num_of_sample
            db = np.sum(dZ, axis=1, keepdims=True) / num_of_sample

            self.__weights[current_weight_key] = current_weight - alpha * dW
            self.__weights[current_bias_key] = current_bias - alpha * db
            if layer_idx > 1:
                if self.activation == "sig":
                    dZ = np.dot(
                        current_weight.T, dZ
                    ) * previous_preds * (1 - previous_preds)
                else:
                    dZ = np.dot(
                        current_weight.T, dZ
                    ) * (1 - previous_preds**2)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neural network
        :param X: The data set
        :param Y: The labels
        :param iterations: The nuber of iteration for the gradient descent
        :param alpha: The learning rate
        :param verbose: Decide weather we display training info or not
        :param graph: Decide weather we display the cost graph
        :param step: The number of step related to verbose
        :return: The evaluation on the data set after n interations.
        """
        _check_iterations(iterations)
        _check_alpha(alpha)
        if graph:
            _check_step(step, iterations)
            cost_list = []
            iter_list = []
        if verbose:
            _check_step(step, iterations)
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, current_cost))
                if graph:
                    cost_list.append(current_cost)
                    iter_list.append(i)
        if graph:
            plt.plot(iter_list, cost_list, "b")
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance to a pickle file
        :param filename: The filename
        :return: Nothing
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a model from a file
        :param filename: Filename
        :return: The loaded model
        """
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
