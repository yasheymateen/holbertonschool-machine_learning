#!/usr/bin/env python3
import numpy as np
""" Valid Convolution"""


def convolve_grayscale_valid(images, kernel):
    """ perform valid convolution on grayscale images """
    num_img = images.shape[0]
    img_row = images.shape[1]
    img_col = images.shape[2]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
    res = np.zeros((num_img, img_row - kernel_row + 1,
                    img_col - kernel_col + 1))
    i = 0
    j = 0
    while(True):
        res[:, j, i] = np.sum(images[:, j:j + kernel_row,
                                     i:i + kernel_col] * kernel, axis=(1, 2))
        if i < img_col - kernel_col:
            i += 1
        elif j < img_row - kernel_row:
            j += 1
            i = 0
        else:
            return res
