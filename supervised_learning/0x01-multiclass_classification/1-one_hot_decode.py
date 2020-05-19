#!/usr/bin/env python3
""" converts one-hot matrix into a vector of labels """


import numpy as np


def one_hot_decode(one_hot):
    """ Decode one-hot coded numpy.ndarray with shape """
    if type(one_hot) is not nd.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
