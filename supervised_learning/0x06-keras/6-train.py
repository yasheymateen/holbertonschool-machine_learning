#!/usr/bin/env python3
""" Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ trains model using mini-batch gradient descent """
    callbacks = []
    if early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(EarlyStopping)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
