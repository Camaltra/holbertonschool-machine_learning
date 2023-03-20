#!/usr/bin/env python3


"""Useless comment"""


def np_slice(matrix, axes: dict):
    """
    Mirror the np slicing
    :param matrix: The given python matrix
    :param axes: Dict of slice axes
    :return: The sliced matrix
    """
    if axes is None:
        axes = {}
    max_axes_idx_range = max(axes) + 1
    slice_idxs = []
    for axe_idx in range(max_axes_idx_range):
        slice_idxs.append(slice(*axes.get(axe_idx) or (None, None, None)))
    return matrix[tuple(slice_idxs)]
