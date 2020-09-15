#!/usr/bin/env python3
""" create a layer with dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates a layer of a neural network using dropout """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=init)
    drop = tf.layers.Dropout(rate=keep_prob)
    return drop(new_layer(prev))
