#!/usr/bin/env python3
""" Early Stopping """
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if gradient descent should stop early """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
        if count >= patience:
            return True, count
        else:
            return False, count
    return False, count
