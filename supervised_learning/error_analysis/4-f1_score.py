#!/usr/bin/env python3

"""useless comments"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Compute the f1 score for all classes
    :param confusion: The confisuion matrix
    :return: The f1 score for all classes
    """
    recall_matrix = sensitivity(confusion)
    precision_matrix = precision(confusion)
    numerator = (recall_matrix * precision_matrix)
    denomiator = (recall_matrix + precision_matrix)
    return 2 * numerator / denomiator
