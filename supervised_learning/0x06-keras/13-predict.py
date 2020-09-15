#!/usr/bin/env python3
""" Predict """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ makes a prediction using neural network """
    if verbose:
        result = network.predict(data, verbose=1)
    else:
        result = network.predict(data, verbose=0)
    return result
