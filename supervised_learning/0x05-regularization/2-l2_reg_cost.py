#!/usr/bin/env python3
""" L2 Regularization Cost """
import tensorflow as tf


def l2_reg_cost(cost):
    """ calculates cost of neural network """
    return cost + tf.losses.get_regularization_losses()
