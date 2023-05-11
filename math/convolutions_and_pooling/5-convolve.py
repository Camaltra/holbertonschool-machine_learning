#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a same convolution on an RGB image w/ multpiple kernel
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel: The kernel operator (Already flipped)
    :param padding: Is either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride:  is a tuple of (sh, sw)
    :return: The output convolve matrix
    """
    m, h, w, ch = images.shape
    kh, kw, _, nc = kernel.shape
    stride_along_height, stride_along_width = stride

    if padding == "valid":
        pad_height, pad_width = 0, 0
    elif padding == "same":
        pad_height = int((((h - 1) * stride_along_height + kh - h) / 2) + 1)
        pad_width = int((((w - 1) * stride_along_width + kw - w) / 2) + 1)
    else:
        pad_height, pad_width = padding

    output_height = int((h - kh + 2 * pad_height) / stride_along_height + 1)
    output_width = int((w - kw + 2 * pad_width) / stride_along_width + 1)

    output = np.zeros(shape=(m, output_height, output_width, ch))

    images_padded = np.pad(
        images,
        [
            (0, 0),
            (pad_height, pad_height),
            (pad_width, pad_width),
            (0, 0)
        ],
        mode="constant"
    )

    for x in range(output_height):
        for y in range(output_width):
            for n in range(ch):
                x_s = x * stride_along_height
                y_s = y * stride_along_width
                sub_image = images_padded[:, x_s:x_s + kh, y_s:y_s + kw, :]
                output[:, x, y, n] = np.sum(
                    (kernel[:, :, :, n] * sub_image),
                    axis=(1, 2, 3)
                )

    return output
