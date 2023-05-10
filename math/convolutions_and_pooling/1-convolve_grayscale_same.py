#!/usr/bin/env python3

"""Useless comment"""

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

    pad_height = int((kh - 1) / 2) if kh % 2 != 0 else int(kh / 2)
    pad_width = int((kw - 1) / 2) if kw % 2 != 0 else int(kw / 2)

    output = np.zeros(shape=(m, h, w))

    images_padded = np.pad(
        images,
        [
            (0, 0),
            (pad_height, pad_height),
            (pad_width, pad_width)
        ],
        mode="constant"
    )

    for x in range(h):
        for y in range(w):
            output[:, x, y] = np.sum(
                (kernel * images_padded[:, x:x + kh, y:y + kw]),
                axis=(1, 2)
            )

    return output
