#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Calculate the new learning rate depend on a decay value and a step
    as the alpha should be update in a step-wise fashion
    :param alpha: The initial learning rate
    :param decay_rate: The decay rate
    :param global_step: The actual step
    :param decay_step: The step-wise indicator
    :return: The new learning rate
    """
    local_decay_value = global_step // decay_step
    return alpha / (1 + local_decay_value * decay_rate)
