#!/usr/bin/env python3

"""Useless comment here"""
import numpy as np
import tensorflow as tf
import os

MODEL_FILEPATH = "cifar10_bis.h5"


def preprocess_data(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_preprocessed = tf.keras.applications.densenet.preprocess_input(X)
    Y_preprocessed = tf.keras.utils.to_categorical(Y)
    return X_preprocessed, Y_preprocessed


def get_bottleneck_features(model: tf.keras.models.Model, input_imgs: np.ndarray) -> tf.keras.dtensor:
    return model.predict(input_imgs, verbose=1)


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

    output = base_inception_v3.layers[-1].output
    inception_v3_feature_extractor = tf.keras.models.Model(inputs=inputs, outputs=output)
    inception_v3_feature_extractor.trainable = False
    for layer in inception_v3_feature_extractor.layers:
        layer.trainable = False

    # Debug
    # for layer in inception_v3_feature_extractor.layers:
        #     print(f"{layer} - {layer.name} - {layer.trainable}")

        # No Idea of how it works
        # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #     rescale=1. / 255,
        #     zoom_range=0.3,
        #     rotation_range=50,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.2,
        #     horizontal_flip=True,
        #     fill_mode='nearest'
        # )
        # augmented_train = train_datagen.flow(x_train, y_train)
        #
        # val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        # augmented_valid = val_datagen.flow(x_train, y_train)

    extracted_inception_feature_train = get_bottleneck_features(inception_v3_feature_extractor, x_train)
    extracted_inception_feature_test = get_bottleneck_features(inception_v3_feature_extractor, x_test)

    decision_model = tf.keras.Sequential()
    decision_model.add(tf.keras.layers.Input(shape=extracted_inception_feature_train.shape[1:]))
    decision_model.add(tf.keras.layers.GlobalAvgPool2D())
    decision_model.add(tf.keras.layers.BatchNormalization())
    decision_model.add(tf.keras.layers.Dense(units=100, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()))
    decision_model.add(tf.keras.layers.BatchNormalization())
    decision_model.add(tf.keras.layers.Dense(units=10, kernel_initializer=tf.keras.initializers.he_normal()))

    decision_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    callback_functions = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_FILEPATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(patience=3)
    ]

    decision_model.fit(x=extracted_inception_feature_train,
                       y=y_train,
                       epochs=100,
                       validation_data=(extracted_inception_feature_test, y_test),
                       callbacks=callback_functions,
                       verbose=1)

    combined_model = tf.keras.models.Model(
        inputs=inception_v3_feature_extractor.inputs,
        outputs=decision_model(inception_v3_feature_extractor.outputs)
    )

    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    combined_model.save(MODEL_FILEPATH)
