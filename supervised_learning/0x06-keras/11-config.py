#!/usr/bin/env python3
""" Save and Load Configuration """
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a models configuration in JSON """
    with open(filename, 'w') as fp:
        fp.write(network.to_json())


def load_config(filename):
    """ loads model with a specific configuration """
    with open(filename, 'r') as fp:
        return K.models.model_from_json(fp.read())
