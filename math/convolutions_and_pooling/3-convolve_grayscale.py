#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a same convolution on grayscale images
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel: The kernel operator (Already flipped)
    :param padding: Is either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride:  is a tuple of (sh, sw)
    :return: The output convolve matrix
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
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

    output = np.zeros(shape=(m, output_height, output_width))

    images_padded = np.pad(
        images,
        [
            (0, 0),
            (pad_height, pad_height),
            (pad_width, pad_width)
        ],
        mode="constant"
    )

    for x in range(output_height):
        for y in range(output_width):
            x_s = x * stride_along_height
            y_s = y * stride_along_width
            sub_image = images_padded[:, x_s:x_s + kh, y_s:y_s + kw]
            output[:, x, y] = np.sum((kernel * sub_image), axis=(1, 2))

    return output
