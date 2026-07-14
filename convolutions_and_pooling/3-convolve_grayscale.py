#!/usr/bin/env python3
"""Module that performs a convolution on grayscale images with
custom padding and stride."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
            grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
            for the convolution
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                images_padded[:, i * sh:i * sh + kh,
                               j * sw:j * sw + kw] * kernel, axis=(1, 2))

    return output
