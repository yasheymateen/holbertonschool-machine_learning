#!/usr/bin/env python3
""" RMSProp Upgraded """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates training operation for nerual network """
    return tf.train.RMSPropOptimizer(alpha, beta2,
                                     epsilon=epsilon).minimize(loss)
