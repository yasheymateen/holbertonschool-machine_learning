#!/usr/bin/env python3
""" calculate loss of network """


import tensorflow as tf


def calculate_loss(y, y_pred):
    """ calculate loss of network """
    return tf.losses.softmax_cross_entropy(y, y_pred)
