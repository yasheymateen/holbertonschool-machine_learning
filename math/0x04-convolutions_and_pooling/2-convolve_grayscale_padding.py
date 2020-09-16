#!/usr/bin/env python3
""" Convolution with Padding """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ function that performs convolution on grayscale images w/pad """
    (ph, pw) = padding
    kh, kw = kernel.shape
    padded_images = np.pad(
        images,
        [(0, 0), (ph, ph), (pw, pw)]
    )
    m, h, w = padded_images.shape
    output_shape = (m, h - kh + 1, w - kw + 1)
    output = np.zeros(output_shape)

    for row in range(output_shape[1]):
        for column in range(output_shape[2]):
            sub_matrix = padded_images[:, row: row + kh, column: column + kw]
            output[:, row, column] = (sub_matrix * kernel).sum(axis=(1, 2))

    return output
