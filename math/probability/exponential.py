#!/usr/bin/env python3


"""Create a class that represente the poisson probability law"""


EXPONENTIAL = 2.7182818285


def get_inverse_list_average(data):
    """
    Get the inverse average of a given list
    :param data: The given list
    :return: The inverse average data
    """
    return len(data) / sum(data)


def _check_data(data):
    """
    Plese this mandatory comment are useless omgßß
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if len(data) < 2:
        raise ValueError("data must contain multiple values")


def _check_lambtha(lambtha):
    """
    Plese this mandatory comment are useless omg
    """
    if lambtha <= 0:
        raise ValueError("lambtha must be a positive value")


class Exponential:
    """
    The class that represente the Exponantial law
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Init methods
        :param data: A list of data to be used to estimate the distribution
        :param lambtha: The expected number of occurences in a given frame
        """
        if data is None:
            _check_lambtha(lambtha)
            self.lambtha = float(lambtha)
        else:
            _check_data(data)
            self.lambtha = len(data)
            self.lambtha = float(get_inverse_list_average(data))

    def pdf(self, x):
        """
        Return the probability mass function for the exponential law
        :param x: The probability need to be checked P(X=x)
        :return: Return 0 if k is out of range, else the result of the pmf
        """
        if x < 0:
            return 0
        return self.lambtha * EXPONENTIAL**-(self.lambtha * x)

    def cdf(self, x):
        """
        Return the cumulative distribution function for the poisson low
        :param x: The probability to be checked such as P(X<=x)
        :return: 0 if k out of range, ele the result of the cdf
        """
        if x < 0:
            return 0
        return 1 - EXPONENTIAL**-(self.lambtha * x)
