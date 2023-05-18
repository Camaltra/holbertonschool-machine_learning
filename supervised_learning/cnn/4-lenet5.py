#!/usr/bin/env python3

"""Useless comment"""


import tensorflow as tf


def lenet5(x, y):
    """
    That builds a modified version of the LeNet-5
    architecture using tensorflow
    :param x: Is a `tf.placeholder` of shape (m, 28, 28, 1)
              containing the input images for the network
    :param y: Is a `tf.placeholder` of shape (m, 10) containing
              the one-hot labels for the network
    :return: A tensor for the softmax activated output
             A training operation that utilizes Adam optimization
                        (with default hyperparameters)
             A tensor for the loss of the netowrk
             A tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    layers = [
        tf.layers.Conv2D(
            filters=6, activation="relu", kernel_size=(5, 5),
            padding="same", kernel_initializer=init,
        ),
        tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.layers.Conv2D(
            filters=16, activation="relu", kernel_size=(5, 5),
            padding="valid", kernel_initializer=init
        ),
        tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.layers.Flatten(),
        tf.layers.Dense(120, activation="relu", kernel_initializer=init),
        tf.layers.Dense(84, activation="relu", kernel_initializer=init),
        tf.layers.Dense(10, kernel_initializer=init),
    ]

    prev_layer_output = x
    for layer in layers:
        prev_layer_output = layer(prev_layer_output)

    loss = tf.losses.softmax_cross_entropy(y, prev_layer_output)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(prev_layer_output, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    output = tf.nn.softmax(prev_layer_output)

    return output, train_op, loss, accuracy
