#!/usr/bin/env python3
""" Calculate the shape of a matrix, returned as a list of integers """


def matrix_shape(matrix):
    if not matrix:
        return None
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
