#!/usr/bin/env python3


"""useless comment"""


import numpy as np


class GaussianProcess:
    """
    Gaussian Process Class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Init the class
        :param X_init: The inputs already sampled with
                       the black-box function
        :param Y_init: The outputs of the black-box
                       function for each input
        :param l: The length parameter for the kernel
        :param sigma_f: The standard deviation given
                        to the output of the black-box
                        function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Compute the Kernel Covariance matrix using RBF
        :param X1: The first Matrix
        :param X2: The second Matrix
        :return: The covariance kernel matrix
        """
        dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist)
