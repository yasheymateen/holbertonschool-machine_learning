#!/usr/bin/env python3
""" Batch Normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes unactivated output of neural network """
    mean = np.sum(Z, axis=0) / Z.shape[0]
    sd = np.sum((Z - mean) ** 2, axis=0) / Z.shape[0]
    Znorm = (Z - mean) / np.sqrt(sd + epsilon)
    return (gamma * Znorm) + beta
