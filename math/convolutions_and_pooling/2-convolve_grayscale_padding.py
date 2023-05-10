#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a same convolution on grayscale images
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel: The kernel operator (Already flipped)
    :param padding: The custom padding
    :return: The output convolve matrix
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_along_height, pad_along_width = padding

    output_height = h - kh + 2 * pad_along_height + 1
    output_width = w - kw + 2 * pad_along_width + 1

    output = np.zeros(shape=(m, output_height, output_width))

    images_padded = np.pad(
        images,
        [
            (0, 0),
            (pad_along_height, pad_along_height),
            (pad_along_width, pad_along_width)
        ],
        mode="constant")

    for x in range(output_height):
        for y in range(output_width):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
