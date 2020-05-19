#!/usr/bin/env python3
""" Create tensor output of previous layer """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ create tf layer """
    initializer = (tf.contrib.layers.
                   variance_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initializer)(prev)
