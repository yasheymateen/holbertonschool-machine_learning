#!/usr/bin/env python3
""" Converts numeric label vector into one-hot matrix """


import numpy as np


def one_hot_encode(Y, classes):
    """ convert numeric label vector """
    try:
        onehot = np.zeros((classes, Y.shape[0]))
        for q, label in enumerate(Y):
            onehot[label][q] = 1
        return onehot
    except Exception:
        return None
