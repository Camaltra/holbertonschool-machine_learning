#!/usr/bin/env python3


"""Create a class that represente the normal probability law"""


EXPONENTIAL = 2.7182818285
PI = 3.1415926536


def get_list_average(data):
    """
    Get the average of a given list
    :param data: The given list
    :return: The average data
    """
    return sum(data) / len(data)


def get_list_stddev(data):
    """
    Get the standard deviation of a given list
    :param data: The given list
    :return: The standart deviation data
    """
    data_mean = get_list_average(data)
    return (sum((x - data_mean)**2 for x in data) / (len(data)))**0.5


def _check_data(data):
    """
    Plese this mandatory comment are useless omgßß
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if len(data) < 2:
        raise ValueError("data must contain multiple values")


def sqrt(x):
    """
    Calculate the sqrt of a num
    :param x:
    :return:
    """
    if x < 0:
        raise ValueError("The number can't be negative")
    return x**0.5


def _check_stddev(stddev):
    """
    Plese this mandatory comment are useless omg
    """
    if stddev <= 0:
        raise ValueError("stddev must be a positive value")


def erf_approx(x):
    """
    Get the erf within an apporoximation
    :param x: The value
    :return: Return an approximation of the erf
    """
    return (2 / sqrt(PI)) * (x - (x**3 / 3) + (x**5 / 10) -
                             (x**7 / 42) + (x**9 / 216))


class Normal:
    """
    The class that represente the Normal law
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Init methods
        :param data: A list of data to be used to estimate the distribution
        :param mean: The mean of the normal law
        :param stddev: The standart deviation of the normal law
        """
        if data is None:
            _check_stddev(stddev)
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            _check_data(data)
            self.mean = float(get_list_average(data))
            self.stddev = float(get_list_stddev(data))

    def z_score(self, x):
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Return the probability mass function for the exponential law
        :param x: The probability need to be checked P(X=x)
        :return: Return 0 if k is out of range, else the result of the pmf
        """
        return (1 / (self.stddev * sqrt(2*PI))) * \
            EXPONENTIAL**(-0.5*self.z_score(x)**2)

    def cdf(self, x):
        """
        Return the cumulative distribution function for the poisson low
        :param x: The probability to be checked such as P(X<=x)
        :return: 0 if k out of range, ele the result of the cdf
        """
        return 0.5 * (1 + erf_approx(self.z_score(x) * sqrt(2)**-1))
