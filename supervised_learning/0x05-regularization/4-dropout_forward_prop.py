#!/usr/bin/env python3
""" Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    cache = {}
    cache['A0'] = X
    for i in range(L):
        keyA = "A{}".format(i + 1)
        keyb = "b{}".format(i + 1)
        keyAo = "A{}".format(i)
        keyW = "W{}".format(i + 1)
        keyD = "D{}".format(i + 1)
        z = (np.matmul(weights[keyW], cache[keyAo]) + weights[keyb])
        if i != L - 1:
            a = np.tanh(z)
            d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            d = np.where(d, 1, 0)
            a_reg /= keep_prob
        else:
            sumatory = np(np.exp(z), axis=0, keepdims=True)
            a_reg = np.exp(z) / sumatory
        cache[keyA] = a_reg

    return cache
