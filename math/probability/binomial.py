#!/usr/bin/env python3


"""Create a class that represente the binomial probability law"""


EXPONENTIAL = 2.7182818285
PI = 3.1415926536


def _check_data(data):
    """
    Plese this mandatory comment are useless omgßß
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if len(data) < 2:
        raise ValueError("data must contain multiple values")


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


def _check_n(n):
    """
    Plese this mandatory comment are useless omg
    """
    if n < 0:
        raise ValueError("n must be a positive value")


def _check_p(p):
    """
    Plese this mandatory comment are useless omg
    """
    if p < 0 or p > 1:
        raise ValueError("p must be greater than 0 and less than 1")


def get_binomial_coefficient(n, k):
    """
    Get the binomial coefficient
    :param n: The number of all event
    :param k: The number of successed event
    :return: The associated coefficient
    """
    return factorial(n) / (factorial(k) * factorial(n - k))


class Binomial:
    """
    The class that represente the Normal law
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Init methods
        :param data: A list of data to be used to estimate the distribution
        :param n: The n value of the binomial law
        :param p: The p value of the binomial law
        """
        if data is None:
            _check_n(n)
            _check_p(p)
            self.n = int(n)
            self.p = float(p)
        else:
            _check_data(data)
            data_mean = sum(data) / len(data)
            data_var = sum([(x - data_mean) ** 2 for
                            x in data]) / (len(data) - 1)
            self.p = 1 - (data_var / data_mean)
            self.n = round(data_mean / self.p)
            self.p = data_mean / self.n

    def pmf(self, k):
        """
        Return the probability mass function for the exponential law
        :param k: The probability need to be checked P(X=k)
        :return: Return 0 if k is out of range, else the result of the pmf
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return get_binomial_coefficient(self.n, k) * self.p**k *\
            (1 - self.p)**(self.n - k)

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
