#!/usr/bin/env python3
""" gradient descent with dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates neural network with dropout regularization """
    weights_copy = weights.copy()
    m = Y.shape[1]
    for i in range(1, L + 1)[::-1]:
        A = cache["A" + str(i)]
        if i == L:
            dZ = A - Y
        else:
            W = weights_copy["W" + str(i + 1)]
            dZ = np.matmul(W.T, dZ) * (1 - (A ** 2))
            dZ *= cache["D" + str(i)]
            dZ /= keep_prob
        dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W = weights_copy["W" + str(i)] - (alpha * dW)
        b = weights_copy["b" + str(i)] - (alpha * db)
        weights["W" + str(i)] = W
        weights["b" + str(i)] = b
