#!/usr/bin/env python3
""" Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ trains model using mini-batch gradient descent """
    def learning_rate_decay(epoch):
        """ learning rate callback """
        alpha_utd = alpha / (1 + (decay_rate * epoch))
        return alpha_utd

    callbacks = []
    if early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(EarlyStopping)
    if learning_rate_decay:
        decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                  verbose=1)
        callbacks.append(decay)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
