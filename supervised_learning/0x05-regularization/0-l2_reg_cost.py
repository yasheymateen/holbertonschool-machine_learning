#!/usr/bin/env python3
"""
calculates cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates cost of network accounting for L2 regularization """
    total = 0
    for i in range(1, L + 1):
        total += np.linalg.norm(weights["W" + str(i)])
    return cost + ((lambtha * total) / (2 * m))
