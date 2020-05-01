#!/usr/bin/env python3
""" Class Poisson that represents poisson distribution """


class Poisson:
    """ Poisson distribution class """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ calculates value of PMF for given number of successes """
        if k < 0:
            return 0
        k = int(k)
        return (pow(self.lambtha, k) *
                pow(2.7182818285, -1 * self.lambtha) /
                factorial(k))

    def cdf(self, k):
        """ Calculates value of the CDF for a given number of successes """
        if k < 0:
            return 0
        k = int(k)
        return sum([self.pmf(n) for n in range(k + 1)])


def factorial(n):
    """ returns the factorial of n """
    if n < 0:
        return None
    if n == 0:
        return 1
    if n < 2:
        return 1
    return n * factorial(n-1)
