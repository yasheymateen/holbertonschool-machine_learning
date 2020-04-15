#!/usr/bin/env python3
""" Calculate the shape of a matrix, returned as a list of integers """
import numpy as np


def matrix_shape(matrix):
    shape = []
    arr1 = np.array(matrix)
    N = arr1.ndim
    if N > 2:
        shape.append(arr1.shape[0])
        shape.append(arr1.shape[1])
        shape.append(shape[0] + shape[1])
    else:
        shape.append(arr1.shape[0])
        shape.append(arr1.shape[1])
    return shape
