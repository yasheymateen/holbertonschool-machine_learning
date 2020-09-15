#!/usr/bin/env python3
""" Gradient Descent with L2 Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases using gradient descent """
    weights_copy = weights.copy()
    m = Y.shape[1]
    for i in range(1, L + 1)[::-1]:
        A = cache["A" + str(i)]
        if i == L:
            dZ = A - Y
        else:
            W = weights_copy["W" + str(i + 1)]
            dZ = np.matmul(W.T, dZ) * (1 - (A ** 2))
        dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dW_reg = dW + ((lambtha / m) * weights_copy["W" + str(i)])
        b = weights_copy["b" + str(i)] - (alpha * db)
        weights["W" + str(i)] = weights_copy["W" + str(i)] - (alpha * dW_reg)
        weights["b" + str(i)] = b
