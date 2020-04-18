#!/usr/bin/env python3
""" function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates using axes with conditionals
    returns new_matrix
    """
    new_matrix = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            new_matrix.append(list(row))
        for row in mat2:
            new_matrix.append(list(row))
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for row, element in zip(mat1, mat2):
            new_matrix.append(list(row) + element)
    return new_matrix
