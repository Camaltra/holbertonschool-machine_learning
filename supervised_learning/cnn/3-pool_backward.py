#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    That performs back propagation over a pooling layer of a neural network
    :param dA: np.ndarray -- containing the partial derivatives with
                             respect to the output of the pooling layer
               m is the number of examples
               h_new is the height of the output
               w_new is the width of the output
               c is the number of channels
    :param A_prev: numpy.ndarray -- containing the output of
                                    the previous layer
                   h_prev is the height of the previous layer
                   w_prev is the width of the previous layer
    :param kernel_shape: tuple -- containing the size of the
                                  kernel for the pooling
                         kh is the kernel height
                         kw is the kernel width
    :param stride: tuple -- containing the strides for
                            the pooling
                   sh is the stride for the height
                   sw is the stride for the width
    :param mode: is a string containing either max or avg, indicating
                 whether to perform maximum or average pooling,
                 respectively
    :return:
    """
    m, h_prev, w_prev, _ = A_prev.shape
    h_stride, w_stride = stride
    kh, kw = kernel_shape
    _, h_new, w_new, c = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for sample in range(m):
        for h_i in range(h_new):
            for w_i in range(w_new):
                for ch in range(c):
                    h_is = h_i * h_stride
                    w_is = w_i * w_stride
                    if mode == "max":
                        window_dA_prev = A_prev[sample, h_is:h_is + kh,
                                                w_is:w_is + kw, ch]
                        mask = (window_dA_prev == np.max(window_dA_prev))
                        dA_prev[sample, h_is:h_is + kh,
                                w_is:w_is + kw, ch] += mask * dA[sample, h_i,
                                                                 w_i, ch]
                    else:
                        mask = dA[sample, h_i, w_i, ch] / (kh * kw)
                        dA_prev[sample,
                                h_is:h_is + kh,
                                w_is:w_is + kw,
                                ch] += np.ones(shape=(kh, kw)) * mask

    return dA_prev
