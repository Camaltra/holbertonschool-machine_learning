#!/usr/bin/env python3

"""useless comments"""

import numpy as np


def specificity(confusion):
    """
    Compute the specificity for all classes
    :param confusion: The confisuion matrix
    :return: The specificity for all classes
    """
    n, _ = confusion.shape
    specificity_matrix = np.zeros(n)

    total_sum = np.sum(confusion)

    for idx in range(n):
        true_positif = confusion[idx][idx]
        false_negatif = np.sum(confusion[idx]) - true_positif
        false_positif = np.sum(confusion[:, idx]) - true_positif

        true_negatif = total_sum - false_positif - false_negatif - true_positif

        specificity_matrix[idx] = true_negatif / (true_negatif + false_positif)

    return specificity_matrix
