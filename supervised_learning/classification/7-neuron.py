#!/usr/bin/env python3


"""Useless comment"""

import numpy as np
import matplotlib.pyplot as plt


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
    def A(self):
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
        a logistic regression model as it use the logistic function
        to make prediction
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

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray,
                         A: np.ndarray, alpha: int = 0.05):
        """
        Make the gradient descent on the neurone
        :param X: The data set
        :param Y: The label set
        :param A: The prediction
        :param alpha: The learning rate
        :return: Nothing, just set the value to private variables
        """
        num_of_sample = self.A.shape[1]
        self.__W = self.W - alpha * np.dot((A - Y), X.T) / num_of_sample
        self.__b = self.b - alpha * np.sum(A - Y) / num_of_sample

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 5000,
              alpha: float = 0.05, verbose: bool = True, graph: bool = True,
              step: int = 100):
        """
        Train a neuron
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
            predictions_matrix = self.forward_prop(X)
            self.gradient_descent(X, Y, predictions_matrix, alpha=alpha)
            if i % step == 0 or i == iterations:
                current_cost = self.cost(Y, self.A)
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
