#!/usr/bin/env python3
""" function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """
    returns new list of coefficients representing derivatives of polynomial
    poly is a list of coefficients representing the polynomial:
    the index of the list represents the power of x that coefficient belongs to
    if poly is not valid return None
    if the derivative is 0 return [0]
    """
    if not type(poly) is list or len(poly) == 0 or type(poly[0]) is not int:
        return None
    _, *poly = poly
    if any(poly):
        derivative = [power * coeff for power, coeff in enumerate(poly, 1)]
        if derivative == 0:
            derivative = [0]
        return derivative
