#!/usr/bin/env python3
""" Binomial distribution class """


class Binomial:
    """ binomial dist class """
    def __init__(self, data=None, n=1, p=0.5):
        """ initialize constructor for binomial class """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            self.p = -1 * (variance / mean - 1)
            n = mean/self.p
            self.n = round(n)
            self.p *= n/self.n

    def pmf(self, k):
        """ calculates PMF for a given number of successes """
        k = int(k)
        if k > self.n or k < 0:
            return 0
        return (factorial(self.n) / factorial(k) / factorial(self.n - k) *
                self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """ calculates value of CDF for given number of successes """
        if k > self.n or k < 0:
            return 0
        sum_cdf = 0
        for i in range(0, int(k) + 1):
            sum_cdf += self.pmf(i)
        return sum_cdf


def factorial(n):
    """ returns factorial of n """
    if n < 0:
        return None
    if n < 2:
        return 1
    return n * factorial(n - 1)
