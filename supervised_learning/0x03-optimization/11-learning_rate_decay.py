#!/usr/bin/env python3
""" Learning Rate Decay """
import numpy as np


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ updates teh learning rate using inverse time decay """
    alpha /= (1 + (decay_rate * (global_step // decay_step)))
    return alpha
