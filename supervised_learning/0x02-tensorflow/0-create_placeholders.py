#!/usr/bin/env python3
""" Create Tensor placeholders for data """

import tensorflow as tf


def create_placeholders(nx, classes):
    """ Create tensor placeholders for input data and one-hot lbls """
    return (tf.placeholder(float, shape=[None, nx], name='x'),
            tf.placeholder(float, shape=[None, classes], name='y'))
