#!/usr/bin/env python3
""" calculates f1 score of confusion matrix """


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates f1 score of confusion matrix """
    sen = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * sen * prec / (sen + prec)
