#!/usr/bin/env python3
""" Input """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds basic neural network with Keras library"""
    kernel_reg = K.regularizers.l2(lambtha)
    list_vars = []
    for i in range(len(layers)):
        if i == 0:
            input_var = K.Input(shape=(nx,))
            hidden = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=kernel_reg)(input_var)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(hidden)
            hidden = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=kernel_reg)(dropout)
    return K.Model(input_var, hidden)
