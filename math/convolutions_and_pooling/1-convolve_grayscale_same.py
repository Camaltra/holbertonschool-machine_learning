#!/usr/bin/env python3

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    :param images: np array shape[num_of_img, height, witdh]
    :param kernel: The kernel operator (Already flipped)
    :return: The output convolve matrix
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_along_height = max((h - 1) + kh - h, 0)
    pad_along_width = max((w - 1) + kw - w, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    output = np.zeros(shape=(m, h, w))

    images_padded = np.zeros(
        shape=(m, h + pad_along_height, w + pad_along_width)
    )
    images_padded[:, pad_top:-pad_bottom, pad_left:-pad_right] = images

    for x in range(h):
        for y in range(w):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
