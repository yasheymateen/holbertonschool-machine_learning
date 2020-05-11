#!/usr/bin/env python3
""" Neuron class defining single neuron performing binary classification """

import numpy as np


class Neuron:
    """ neuron class """

    def __init__(self, nx):
        """ nx: number of input features to the neuron """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Returns weights """
        return self.__W

    @property
    def b(self):
        """ Returns bias """
        return self.__b

    @property
    def A(self):
        """ Returns activation values """
        return self.__A

    def forward_prop(self, X):
        """ calculates forward propagation of neuron """
        self.__A = 1 / (1 + np.exp(-1 * (np.dot(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """ calculates cost of the model using logistic regression """
        costsum = 0
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """ Evalutes the neuron's predictions """
        A = np.ndarray((1, X.shape[1]))
        A[0] = self.forward_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)
