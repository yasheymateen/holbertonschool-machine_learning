#!/usr/bin/env python3
""" function that calculates summation of i^2 up to n """


def summation_i_squared(n):
    """ function that returns integer value of sum """
    if type(n) is not int or n <= 0:
        return None
    return sum(map(lambda i: i ** 2, range(1, n + 1))) or None
