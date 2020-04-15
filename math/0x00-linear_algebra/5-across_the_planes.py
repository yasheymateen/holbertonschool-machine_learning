#!/usr/bin/env python3
""" Implemnt function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    matrix_sum = []

    for row_first, row_second in zip(mat1, mat2):
        matrix_sum.append([])
        for i, j in zip(row_first, row_second):
            matrix_sum[-1].append(i + j)
    return matrix_sum
