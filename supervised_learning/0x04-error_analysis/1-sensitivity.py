#!/usr/bin/env python3
""" calculates sensitivity of confusion matrix """


import numpy as np


def sensitivity(confusion):
    """ calculates sensitivity of confusion matrix"""
    return np.asarray([confusion[row][row] / confusion[row].sum()
                       for row in range(confusion.shape[0])])
