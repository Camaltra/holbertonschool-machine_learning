#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def conv_backward(dZ, A_prev, W, _, padding="same", stride=(1, 1)):
    """

    :param dZ: np.ndarray -- containing the partial derivatives with
                             respect to the unactivated output of the
                             convolutional layer
               m is the number of examples
               h_new is the height of the output
               w_new is the width of the output
               c_new is the number of channels in the output
    :param A_prev: np.ndarray -- containing the output of the previous
                                 layer
                   h_prev is the height of the previous layer
                   w_prev is the width of the previous layer
                   c_prev is the number of channels in the previous layer
    :param W: np.ndarray -- containing the kernels for the convolution
              kh is the filter height
              kw is the filter width
    :param _: Anything
    :param padding: Is a string that is either same or valid, indicating
                    the type of padding used
    :param stride: np.ndarray -- containing the strides for the convolution
                   sh is the stride for the height
                   sw is the stride for the width
    :return: The partial derivatives with respect to the previous layer
             (dA_prev), the kernels (dW), and the biases (db),
             respectively
    """
    m, h_prev, w_prev, ch_prev = A_prev.shape
    h_stride, w_stride = stride
    kh, kw, _, _ = W.shape
    _, h_new, w_new, c_new = dZ.shape

    if padding == "same":
        h_pad = int(np.ceil(((h_prev - 1) * h_stride + kh - h_prev) / 2))
        w_pad = int(np.ceil(((w_prev - 1) * w_stride + kw - w_prev) / 2))
    else:
        h_pad, w_pad = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        [(0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)],
        mode="constant"
    )
    dA_padded = np.zeros(shape=A_prev_padded.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for sample in range(m):
        for h_i in range(h_new):
            for w_i in range(w_new):
                for ch in range(c_new):
                    h_is = h_i * h_stride
                    w_is = w_i * w_stride
                    dA_padded[sample,
                              h_is:h_is + kh,
                              w_is:w_is + kw,
                              :] += W[:, :, :, ch] * dZ[sample, h_i, w_i, ch]
                    dW[:, :, :, ch] += A_prev_padded[sample,
                                                     h_is:h_is + kh,
                                                     w_is:w_is + kw,
                                                     :] * dZ[sample, h_i,
                                                             w_i, ch]

    if padding == "same":
        dA = dA_padded[:, h_pad:-h_pad, w_pad:-w_pad, :]
    else:
        dA = dA_padded

    return dA, dW, db
