#!/usr/bin/env python3
""" function that  calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ calculates integral and returns list of coefficients """
    if not all(type(C) in (float, int) for c in poly) or type(C) != int:
        return None
    integral = [c/i if c % i != 0 else c//i for i, c in enumerate(poly, 1)]
    while len(integral) > 0 and integral[-1] == 0:
        integral.pop()
    return [C] + integral
