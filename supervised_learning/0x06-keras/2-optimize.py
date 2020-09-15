#!/usr/bin/env python3
""" Optimize """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ sets up Adam optimization for a keras model """
    Adam = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=Adam,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
