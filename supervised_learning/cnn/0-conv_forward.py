#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    :param A_prev: np.ndarray -- containing the output of the previous
                                 layer
                   h_prev is the height of the previous layer
                   w_prev is the width of the previous layer
                   c_prev is the number of channels in the previous layer
    :param W: np.ndarray -- containing the kernels for the convolution
              kh is the filter height
              kw is the filter width
    :param b: np.ndarray -- (1, 1, 1, c_new) containing the biases
                            applied to the convolution
    :param activation: Is an activation function applied to the convolution
    :param padding: Is a string that is either same or valid, indicating
                    the type of padding used
    :param stride: np.ndarray -- containing the strides for the convolution
                   sh is the stride for the height
                   sw is the stride for the width
    :return: The output of the convolutional layer
    """
    m, h_prev, w_prev, ch_prev = A_prev.shape
    h_stride, w_stride = stride
    kh, kw, _, ch = W.shape

    if padding == "valid":
        h_pad = 0
        w_pad = 0
    else:
        h_pad = int(np.ceil(((h_prev - 1) * h_stride + kh - h_prev) / 2))
        w_pad = int(np.ceil(((w_prev - 1) * w_stride + kw - w_prev) / 2))

    h_output = int(((2 * h_pad - kh + h_prev) / h_stride) + 1)
    w_output = int(((2 * w_pad - kw + w_prev) / w_stride) + 1)

    output = np.zeros(shape=(m, h_output, w_output, ch))

    A_prev_padded = np.pad(
        A_prev,
        [(0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)],
        mode="constant"
    )

    for h_i in range(h_output):
        for w_i in range(w_output):
            for ch_i in range(ch):
                x_s = h_i * h_stride
                y_s = w_i * w_stride
                sub_image = A_prev_padded[:, x_s:x_s + kh, y_s:y_s + kw, :]
                output[:,
                       h_i,
                       w_i,
                       ch_i] = np.sum(sub_image * W[:, :,
                                                    :, ch_i], axis=(1, 2, 3))

    return activation(output + b)
