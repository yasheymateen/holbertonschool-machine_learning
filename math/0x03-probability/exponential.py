#!/usr/bin/env python3
""" Exponential distribution class """


class Exponential:
    """ Exponential class """
    def __init__(self, data=None, lambtha=1.):
        """ initial class constructor """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """ Return pdf of exp distribution of x """
        if x < 0:
            return 0
        return self.lambtha * pow(2.7182818285, -1 *
                                  self.lambtha * x)

    def cdf(self, x):
        """ Return cdf of exp distribution of x """
        if x < 0:
            return 0
        return 1 - pow(2.7182818285, -1 * self.lambtha * x)
