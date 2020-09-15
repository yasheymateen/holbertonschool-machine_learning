#!/usr/bin/env python3
""" updates variable w/ gradient descent """
import numpy as np


def update_variable_momentum(alpha, beta1, var, grad, v):
    """ udpates variable using gradient descent """
    dV = (beta1 * v) + ((1 - beta1) * grad)
    var = var - (alpha * dV)
    return var, dV
