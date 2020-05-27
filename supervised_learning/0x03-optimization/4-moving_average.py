#!/usr/bin/env python3
""" calculate weighted moving average of data set """


def moving_average(data, beta):
    """ calculate weighted moving average of data set """
    temp = [0]
    unbiased = []
    for idx, dt in enumerate(data):
        temp.append(beta * temp[idx] + (1 - beta) * dt)
        unbiased.append(temp[idx + 1] / (1 - beta ** (idx + 1)))
    return unbiased
