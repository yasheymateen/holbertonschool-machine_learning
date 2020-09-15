#!/usr/bin/env python3
""" Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """ saves the entire model """
    network.save(filename)


def load_model(filename):
    """ loads the entire model """
    return K.models.load_model(filename)
