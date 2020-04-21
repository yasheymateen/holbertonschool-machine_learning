#!/usr/bin/env python3
""" function that slices a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """ Slices numpy.ndarray along axes """
    piece = (slice(*axes.get(depth, (None, None)))
             for depth in range(len(matrix.shape)))
    return matrix[tuple(piece)]
