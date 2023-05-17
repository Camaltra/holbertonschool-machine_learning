#!/usr/bin/env python3

"""Useless comment"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    :param A_prev: np.ndarray -- containing the output of the previous
                                 layer
                   h_prev is the height of the previous layer
                   w_prev is the width of the previous layer
                   c_prev is the number of channels in the previous layer
    :param kernel_shape: tuple -- containing the size of the
                                  kernel for the pooling
                         kh is the kernel height
                         kw is the kernel width
    :param stride: np.ndarray -- containing the strides for the convolution
                   sh is the stride for the height
                   sw is the stride for the width
    :param mode: is a string containing either max or avg, indicating
                 whether to perform maximum or average pooling,
                 respectively
    :return: The output of the pooling layer
    """
    m, h_prev, w_prev, ch_prev = A_prev.shape
    h_stride, w_stride = stride
    kh, kw = kernel_shape

    h_output = int((h_prev - kh) / h_stride + 1)
    w_output = int((w_prev - kw) / w_stride + 1)

    output = np.zeros(shape=(m, h_output, w_output, ch_prev))

    for h_i in range(h_output):
        for w_i in range(w_output):
            x_s = h_i * h_stride
            y_s = w_i * w_stride
            sub_img = A_prev[:, x_s:x_s + kh, y_s:y_s + kw, :]
            if mode == "max":
                output[:, h_i, w_i, :] = np.max(sub_img, axis=(1, 2))
            elif mode == "avg":
                output[:, h_i, w_i, :] = np.mean(sub_img, axis=(1, 2))

    return output
