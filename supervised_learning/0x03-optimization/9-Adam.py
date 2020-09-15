#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """ updates a variable in place """
    dV = (beta1 * v) + ((1 - beta1) * grad)
    dS = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    dV_corr = dV / (1 - (beta1 ** t))
    dS_corr = dS / (1 - (beta2 ** t))
    var = var - (alpha * (dV_corr / (np.sqrt(dS_corr) + epsilon)))
    return var, dV, dS
