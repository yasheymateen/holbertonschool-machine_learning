#!/usr/bin/env python3
""" Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    def learning_rate_decay(epoch):
        """ learning rate callback"""
        alpha_utd = alpha / (1 + (decay_rate * epoch))
        return alpha_utd

    callbacks = []
    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 mode='min')
        callbacks.append(checkpoint)
    if validation_data and learning_rate_decay:
        decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                  verbose=1)
        callbacks.append(decay)
    if validation_data and early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience,
                                                  monitor='val_loss',
                                                  mode='min')
        callbacks.append(EarlyStopping)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
