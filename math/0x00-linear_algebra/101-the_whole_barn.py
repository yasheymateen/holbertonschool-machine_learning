#!/usr/bin/env python3
""" function that adds two matrices """


def add_matrices(mat1, mat2):
    """ Performs element wise operations of matrix addition """
    try:
        return add_matrices_map(mat1, mat2)
    except ValueError:
        return None


def add_matrices_map(mat1, mat2):
    """ Recursively map matricces to add """
    if len(mat1) != len(mat2):
        raise ValueError
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        return list(map(add_matrices_map, mat1, mat2))
    return [i + j for i, j in zip(mat1, mat2)]
