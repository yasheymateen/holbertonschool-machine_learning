#!/usr/bin/env python3
""" calculates the specificity of a confusion matrix """


import numpy as np


def specificity(confusion):
    """ calculate specificity of confusion matrix """
    return np.asarray([np.delete(np.delete(confusion, row, 1), row, 0).sum()
                       / np.delete(confusion, row, 0).sum()
                       for row in range(confusion.shape[0])])
