#!/usr/bin/env python3

"""Useless comment"""
import tensorflow as tf
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Create the ResNet50 Model
    :return: The ResNet50 model
    """
    init = tf.keras.initializers.he_normal()
    input = tf.keras.Input(shape=(224, 224, 3))

    conv_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(7, 7),
                                    strides=(2, 2),
                                    padding="same",
                                    kernel_initializer=init)(input)
    norm_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_1)
    act_1 = tf.keras.layers.ReLU()(norm_1)

    max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                              strides=(2, 2),
                                              padding="same")(act_1)

    pr_block_1_2x = projection_block(max_pool_1, [64, 64, 256], s=1)
    id_block_1_2x = identity_block(pr_block_1_2x, [64, 64, 256])
    id_block_2_2x = identity_block(id_block_1_2x, [64, 64, 256])

    pr_block_1_3x = projection_block(id_block_2_2x, [128, 128, 512])
    id_block_1_3x = identity_block(pr_block_1_3x, [128, 128, 512])
    id_block_2_3x = identity_block(id_block_1_3x, [128, 128, 512])
    id_block_3_3x = identity_block(id_block_2_3x, [128, 128, 512])

    pr_block_1_4x = projection_block(id_block_3_3x, [256, 256, 1024])
    id_block_1_4x = identity_block(pr_block_1_4x, [256, 256, 1024])
    id_block_2_4x = identity_block(id_block_1_4x, [256, 256, 1024])
    id_block_3_4x = identity_block(id_block_2_4x, [256, 256, 1024])
    id_block_4_4x = identity_block(id_block_3_4x, [256, 256, 1024])
    id_block_5_4x = identity_block(id_block_4_4x, [256, 256, 1024])

    pr_block_1_5x = projection_block(id_block_5_4x, [512, 512, 2048])
    id_block_1_5x = identity_block(pr_block_1_5x, [512, 512, 2048])
    id_block_2_5x = identity_block(id_block_1_5x, [512, 512, 2048])

    max_pool = tf.keras.layers.AvgPool2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding="valid")(id_block_2_5x)

    dense_output = tf.keras.layers.Dense(units=1000,
                                         activation="softmax",
                                         kernel_initializer=init)(max_pool)

    return tf.keras.models.Model(inputs=input, outputs=dense_output)
