#!/usr/bin/env python3
""" Adam Upgraded """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates the training operation for neural network """
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
