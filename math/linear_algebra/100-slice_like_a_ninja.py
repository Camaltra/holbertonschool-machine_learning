#!/usr/bin/env python3


"""Useless comment"""


def np_slice(matrix, axes: dict[int, int | None] | None):
    if axes is None:
        axes = {}
    max_axes_idx_range = max(axes) + 1
    slice_idxs = []
    for axe_idx in range(max_axes_idx_range):
        slice_idxs.append(slice(*axes.get(axe_idx) or (None, None, None)))
    return matrix[tuple(slice_idxs)]
