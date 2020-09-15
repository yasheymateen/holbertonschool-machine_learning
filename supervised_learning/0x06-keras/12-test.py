#!/usr/bin/env python3
""" Test """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ test a neural network """
    if verbose:
        result = network.evaluate(data, labels, verbose=1)
    else:
        result = network.evaluate(data, labels, verbose=0)
    return result
