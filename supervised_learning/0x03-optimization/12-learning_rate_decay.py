#!/usr/bin/env python3
""" Learning Rate Decay """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ creates learning rate decay operation """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
