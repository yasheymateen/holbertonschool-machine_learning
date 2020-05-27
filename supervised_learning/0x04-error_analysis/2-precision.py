#!/usr/bin/env python3
""" calculates precision of confusion matrix """


import numpy as np


def precision(confusion):
    """ calculates precision of confusion matrix """
    return np.asarray([confusion[row][row] / confusion[:, row].sum()
                       for row in range(confusion.shape[0])])
