#!/usr/bin/env python3
""" function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """
    function that adds arrays
    where matrix_sum  is matrix to return
    """
    if len(arr1) != len(arr2):
        return None

    matrix_sum = []
    for i, j in zip(arr1, arr2):
        matrix_sum.append(i + j)
    return matrix_sum
