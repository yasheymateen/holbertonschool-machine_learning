#!/usr/bin/env python3
""" Deep Neural Network Class """


import numpy as np


class DeepNeuralNetwork:
    """ deep neural network class """

    def __init__(self, nx, layers):
        """ nx: number of input features """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if t ype(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {"W1": np.random.randn(layers[0], nx) *
                          np.sqrt(2 / nx),
                          "b1": np.zeros((layers[0], 1))}
        for layer, size in enumerate(layers[1:], 2):
            cur = "W" + str(layer)
            self.__weights[cur] = (np.random.randn(size, layers[layer - 2]) *
                                   np.sqrt(2 / layers[layer - 2]))
            cur = "b" + str(layer)
            self.__weights[cur] = np.zeros((layers[layer - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ forward propagation of neural network """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            curw = "W" + str(layer + 1)
            curb = "b" + str(layer + 1)
            cura = "A" + str(layer + 1)
            preva = "A" + str(layer)
            z = (np.dot(self.__weights[curw], self.__cache[preva]) +
                 self.__weights[curb])
            self.__cache[cura] = 1 / (1 + np.exp(-z))
        return self.__cache["A" + str(self.__L)], self.__cache
