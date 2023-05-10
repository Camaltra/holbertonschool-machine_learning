#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a same convolution on grayscale images
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel: The kernel operator (Already flipped)
    :return: The output convolve matrix
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_height = h - kh + 1
    output_width = w - kw + 1

    output = np.zeros(shape=(m, output_height, output_width))

    for x in range(output_height):
        for y in range(output_width):
            output[:, x, y] = np.sum(
                (kernel * images[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
