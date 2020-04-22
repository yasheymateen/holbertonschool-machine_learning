#!/usr/bin/env python3
""" function that concatenates two matrices along a specific axes """


def matrix_shape(matrix):
    """ gives shape of matrix """
    if not matrix:
        return None
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])


def cat(mat1, mat2, result, shape, axis, dim):
    """ recursively concatenate"""
    if axis == dim:
        result.extend(mat1)
        result.extend(mat2)
        return
    elif axis < dim:
        for x in range(shape[axis]):
            k = []
            cat(mat1[x], mat2[x], k, shape, axis + 1, dim)
            result.append(k)


def cat_matrices(mat1, mat2, axis=0):
    """ concatenates the matrices"""
    result = []
    m1_s = matrix_shape(mat1)
    m2_s = matrix_shape(mat2)

    if len(m1_s) <= axis or len(m2_s) <= axis:
        return None
    if len(m1_s) != len(m2_s):
        return None
    if m1_s[:axis] != m2_s[:axis] or m1_s[axis+1:] != m2_s[axis+1:]:
        return None
    cat(mat1, mat2, result, m1_s, 0, axis)
    return result
