#!/usr/bin/env python3
""" Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ creates tf.layer including l2 regulatization """
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=init,
                                kernel_regularizer=regularizer)
    return new_layer(prev)
