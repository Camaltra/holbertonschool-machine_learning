#!/usr/bin/env python3


"""useless comment"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Compute the early stoping given parameter
    :param cost: The current validation cost of the neural network
    :param opt_cost: The lowest recorded validation cost of the neural network
    :param threshold: The threshold used for early stopping
    :param patience: The patience count used for early stopping
    :param count: The count of how long the threshold has not been met
    :return: A boolean of whether the network should be stopped early,
             followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
