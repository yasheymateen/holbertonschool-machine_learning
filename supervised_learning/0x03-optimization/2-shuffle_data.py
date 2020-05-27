#!/usr/bin/env python3
"""Shuffle data in two matrices in the same way"""


import numpy as np


def shuffle_data(X, Y):
    """Shuffle data in two matrices"""
    shufflid = np.random.permutation(X.shape[0])
    return X[shufflid], Y[shufflid]
