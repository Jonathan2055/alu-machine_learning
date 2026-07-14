#!/usr/bin/env python3
"""Module that performs pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing
            multiple images
        kernel_shape: tuple of (kh, kw) containing the kernel shape
            for the pooling
        stride: tuple of (sh, sw)
        mode: type of pooling, either 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    if mode == 'max':
        op = np.max
    elif mode == 'avg':
        op = np.mean

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j, :] = op(
                images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                axis=(1, 2))

    return output
