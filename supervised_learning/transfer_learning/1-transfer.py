#!/usr/bin/env python3

"""Useless comment here"""
import numpy as np
import tensorflow as tf
import os

MODEL_FILEPATH = "cifar10_1.h5"


def preprocess_data(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_preprocessed = tf.keras.applications.densenet.preprocess_input(X)
    Y_preprocessed = tf.keras.utils.to_categorical(Y)
    return X_preprocessed, Y_preprocessed


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    model_used = "densenet_121"

    resize_layer = tf.keras.layers.Lambda(
        lambda image: tf.image.resize(image, (130, 130)))(inputs)

    base_inception_v3 = tf.keras.applications.DenseNet121(
        weights="imagenet",
        input_tensor=resize_layer,
        input_shape=(130, 130, 3),
        include_top=False,  # Delete the predictions layer, as well as the GlobalAveragePooling one
    )

    # base_inception_v3.summary()

    output = base_inception_v3.layers[-1].output
    inception_v3_feature_extractor = tf.keras.models.Model(inputs=inputs, outputs=output)
    for layer in inception_v3_feature_extractor.layers:
        if "block16" in layer.name or layer.name in ["bn", "relu"]:
            continue
        layer.trainable = False

    decision_model = tf.keras.Sequential()
    decision_model.add(inception_v3_feature_extractor)
    decision_model.add(tf.keras.layers.GlobalAvgPool2D())
    decision_model.add(tf.keras.layers.Dense(units=10, kernel_initializer=tf.keras.initializers.he_normal()))

    decision_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.legacy.Adam(),
        metrics=['accuracy']
    )

    callback_functions = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_FILEPATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ]

    decision_model.fit(x=x_train,
                       y=y_train,
                       epochs=20,
                       validation_data=(x_test, y_test),
                       callbacks=callback_functions,
                       verbose=1)

