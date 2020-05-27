#!/usr/bin/env python3
""" normalize input data """


import numpy as np


def normalize(X, m, s):
    """ normalize data """
    return (X - m) / s
