#!/usr/bin/env python3
""" RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates var using RMSProp optimization algrotithm """
    dV = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var = var - (alpha * (grad / (np.sqrt(dV) + epsilon)))
    return var, dV
