#!/usr/bin/env python3
""" Module to perform same convolution to grayscale images """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs same convolution grayscale images """
    kh, kw = kernel.shape
    m, h, w = images.shape
    pw = int(kw / 2 if kw % 2 == 0 else (kw - 1) / 2)
    ph = int(kh / 2 if kh % 2 == 0 else (kh - 1) / 2)
    padded_images = np.pad(
        images,
        [(0, 0), (ph, ph), (pw, pw)]
    )
    output = np.zeros(images.shape)

    for row in range(h):
        for column in range(w):
            sub_matrix = padded_images[:, row: row + kh, column: column + kw]
            output[:, row, column] = (sub_matrix * kernel).sum(axis=(1, 2))

    return output
