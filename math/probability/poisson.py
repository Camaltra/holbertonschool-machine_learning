#!/usr/bin/env python3


"""Create a class that represente the poisson probability law"""


EXPONENTIAL = 2.7182818285


def get_list_average(data):
    """
    Get the average of a given list
    :param data: The given list
    :return: The average data
    """
    return sum(data) / len(data)


def factorial(number):
    """
    Return the factirial for a given number
    :param number: The given number
    :return: The factorial
    """
    factorial_result = 1
    for i in range(1, number+1):
        factorial_result *= i
    return factorial_result


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


class Poisson:
    """
    The class that represente the Poisson law
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
            self.lambtha = float(get_list_average(data))

    def pmf(self, k):
        """
        Return the probability mass function for the poisson law
        :param k: The probability need to be checked P(X=k)
        :return: Return 0 if k is out of range, else the result of the pmf
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return (EXPONENTIAL**(-self.lambtha) * self.lambtha**k) / factorial(k)

    def cdf(self, k):
        """
        Return the cumulative distribution function for the poisson low
        :param k: The probability to be checked such as P(X<=k)
        :return: 0 if k out of range, ele the result of the cdf
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf_result = 0
        for sub_k in range(k + 1):
            cdf_result += self.pmf(sub_k)
        return cdf_result
