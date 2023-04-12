#!/usr/bin/env python3

"""Useless comment"""

import numpy as np
import matplotlib.pyplot as plt


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
        self.__W1 = np.random.normal(size=(self.nodes, self.nx))
        self.__b1 = np.zeros((self.nodes, 1))
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
    def A2(self) -> np.ndarray:
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

    def _compute_forward_prop_layer(self, inputs, weights, bias):
        """
        Compute a layer of the neural network
        :param inputs: The data inputs
        :param weights: The weight of the current layer
        :param bias: The bias of the current layer
        :return: The output data of the layer
        """
        pre_processed_data = np.dot(weights, inputs) + bias
        return np.vectorize(
            lambda x: 1 / (1 + np.exp(-x))
        )(pre_processed_data)

    def forward_prop(self, X):
        """
        Compute the forward prop of the neural network of 2 layers
        Not the best ways to code it, but a good one to visualize
        :param X: The data fron the input layer
        :return: The output data of the neural network
        """
        layer_one = {"inputs": X, "weights": self.W1, "bias": self.b1}
        self.__A1 = self._compute_forward_prop_layer(**layer_one)
        layer_two = {"inputs": self.A1, "weights": self.W2, "bias": self.b2}
        self.__A2 = self._compute_forward_prop_layer(**layer_two)
        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Get the cost result of the neural network
        :param Y: The thruht label
        :param A: The predict value
        :return: The cost of the neural network
        """
        num_of_sample = A.shape[1]
        log_loss_res = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return -num_of_sample ** -1 * np.sum(log_loss_res)

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        """
        Evaluate the prediction made by the neural network
        :param X: The data set
        :param Y: The thruth labels
        :return: The predictions and the cost
        """
        self.forward_prop(X)
        preds = np.where(self.A2 < 0.5, 0, 1)
        return preds, self.cost(Y, self.A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform the gradient descent on the neural network
        :param X: The data set
        :param Y: The thruth label
        :param A1: The output matrix of the first layer
        :param A2: The output matrix for the second layer
        :param alpha: The learning rate
        :return: Nothing, just compute the operation
        """
        num_of_sample = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / num_of_sample
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_sample

        dZ1 = np.dot(self.W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / num_of_sample
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_sample

        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = self.b2 - alpha * db2
        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 5000,
              alpha: float = 0.05, verbose: bool = True, graph: bool = True,
              step: int = 100):
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
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, self.A2)
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
