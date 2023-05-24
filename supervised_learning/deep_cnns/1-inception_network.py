#!/usr/bin/env python3

"""Useless comment"""

import tensorflow as tf
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Create the GoogleNet model
    :return: The GoogleNet model
    """
    x = tf.keras.Input(shape=(224, 224, 3))

    conv_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(7, 7),
                                    strides=(2, 2),
                                    padding="same",
                                    activation="relu")(x)
    max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                              strides=(2, 2),
                                              padding="same")(conv_1)

    conv_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(1, 1),
                                    padding="same",
                                    activation="relu")(max_pool_1)
    conv_3 = tf.keras.layers.Conv2D(filters=192,
                                    kernel_size=(3, 3),
                                    padding="same",
                                    activation="relu")(conv_2)
    max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                              strides=(2, 2),
                                              padding="same")(conv_3)

    inception_2a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inception_2b = inception_block(inception_2a, [128, 128, 192, 32, 96, 64])

    max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                              strides=(2, 2),
                                              padding="same")(inception_2b)

    inception_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    max_pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                              strides=(2, 2),
                                              padding="same")(inception_4e)

    inception_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    avg_pooling = tf.keras.layers.AvgPool2D(pool_size=(7, 7),
                                            padding="same")(inception_5b)

    dropout = tf.keras.layers.Dropout(rate=0.4)(avg_pooling)

    fc_1 = tf.keras.layers.Dense(units=1000, activation="softmax")(dropout)

    return tf.keras.models.Model(inputs=x, outputs=fc_1)
