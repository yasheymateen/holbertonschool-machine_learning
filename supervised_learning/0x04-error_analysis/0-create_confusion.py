#!/usr/bin/env python3
""" create confusion matrix """


import numpy as np


def create_confusion_matrix(labels, logits):
    """ create confusion matrix """
    return np.dot(labels.T, logits)
