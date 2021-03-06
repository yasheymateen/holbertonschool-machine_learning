#!/usr/bin/env python3
""" LeNet-5 (Tensorflow) """
import tensorflow as tf


def lenet5(x, y):
    """ builds a modified version of LeNet-5 architecture using tf """
    he_normal = tf.contrib.layers.variance_scaling_initializer()
    relu = tf.nn.relu
    conv_lay1 = tf.layers.Conv2D(filters=6,
                                 kernel_size=(5, 5),
                                 padding='same',
                                 activation=relu,
                                 kernel_initializer=he_normal)(x)
    pool_lay1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(conv_lay1)
    conv_lay2 = tf.layers.Conv2D(filters=16,
                                 kernel_size=(5, 5),
                                 padding='valid',
                                 activation=relu,
                                 kernel_initializer=he_normal)(pool_lay1)
    pool_lay2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(conv_lay2)
    flatten = tf.layers.Flatten()(pool_lay2)
    full_lay3 = tf.layers.Dense(units=120,
                                activation=relu,
                                kernel_initializer=he_normal)(flatten)
    full_lay4 = tf.layers.Dense(units=84,
                                kernel_initializer=he_normal)(full_lay3)
    softmax = tf.layers.Dense(units=10,
                              kernel_initializer=he_normal)(full_lay4)
    loss = tf.losses.softmax_cross_entropy(y, logits=softmax)
    y_max = tf.argmax(y, axis=1)
    y_pred_max = tf.argmax(activated_sof, axis=1)
    bias = tf.cast(tf.equal(y_max, y_pred_max), dtype=tf.float32)
    accuracy = tf.reduce_mean(bias)
    train = tf.train.AdamOptimizer().minimize(loss)
    activated_sof = tf.nn.softmax(softmax)

    return activated_sof, train, loss, accuracy
