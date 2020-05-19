#!/usr/bin/env python3
""" create training op for network """


import tensorflow as tf


def create_train_op(loss, alpha):
    """ create training op for network """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
