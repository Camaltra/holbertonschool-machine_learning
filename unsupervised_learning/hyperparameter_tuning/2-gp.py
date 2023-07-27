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

    def predict(self, X_s):
        """
        Compute the mean and the variance for new datapoint
        following the GP
        :param X_s: The new datapoint
        :return: The mean and the std for theses new points
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(self.K)

        mu_s = np.matmul(np.matmul(K_s.T, K_inv), self.Y)
        covariance_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)

        return mu_s.reshape(1, -1), np.diagonal(covariance_s)

    def update(self, X_new, Y_new):
        """
        Update the value with the news predicted
        :param X_new: The new X value
        :param Y_new: The computed Y value from the X value
        :return: Nothing, only update the variables
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
