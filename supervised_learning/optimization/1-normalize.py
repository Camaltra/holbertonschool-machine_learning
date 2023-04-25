#!/usr/bin/env python3

"""Useless comment"""


def normalize(X, mean, std):
    """
    Normalize a matrix given for each column
    the mean and the std
    :param X: The data set
    :param mean: The mean for each column
    :param std: The std for each column
    :return:
    """
    return (X - mean) / std
