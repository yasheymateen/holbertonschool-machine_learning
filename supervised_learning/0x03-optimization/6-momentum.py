#!/usr/bin/env python3
""" Momentum Upgraded """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates training operation for neural network """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
