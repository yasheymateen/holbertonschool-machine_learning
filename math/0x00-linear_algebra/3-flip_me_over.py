#!/usr/bin/env python3
"""transpose a 2D matrix"""


def matrix_transpose(matrix):
    """
    function that transposes matrix using nested for loops
    and enumerate to append to new matrix
    """
    matrix1 = []
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            if i == 0:
                matrix1.append([])
            matrix1[j].append(col)
    return matrix1
