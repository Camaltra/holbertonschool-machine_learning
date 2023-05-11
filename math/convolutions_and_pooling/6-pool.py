#!/usr/bin/env python3

"""Useless comment"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform a pooling on a given image
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel_shape: The kernel shape (Tuple)
    :param stride:  is a tuple of (sh, sw)
    :param mode: The mode of the pooling
    :return: The output convolve matrix
    """
    m, h, w, ch = images.shape
    kh, kw = kernel_shape
    stride_along_height, stride_along_width = stride

    output_height = int((h - kh) / stride_along_height + 1)
    output_width = int((w - kw) / stride_along_width + 1)

    output = np.zeros(shape=(m, output_height, output_width, ch))

    for x in range(output_height):
        for y in range(output_width):
            x_s = x * stride_along_height
            y_s = y * stride_along_width
            sub_image = images[:, x_s:x_s + kh, y_s:y_s + kw, :]
            if mode == 'max':
                output[:, x, y, :] = sub_image.max(axis=(1, 2))
            else:
                output[:, x, y, :] = np.mean(sub_image, axis=(1, 2))

    return output
