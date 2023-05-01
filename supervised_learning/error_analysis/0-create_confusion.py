#!/usr/bin/env python3

"""useless comments"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create the confusion matrix
    :param labels: The thruth label in one hot array
    :param logits: The predicted values in one hot array
    :return: The confusion matrix
    """
    print(labels)
    m, n = labels.shape
    conf_matrix = np.zeros((n, n))
    for idx in range(m):
        conf_matrix[np.argmax(labels[idx])][np.argmax(logits[idx])] += 1
    return conf_matrix
