#!/usr/bin/env python3


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build vanilla autoencoder
    :param input_dims: Input dimensions
    :param hidden_layers: Hidden layers unit values
    :param latent_dims: Latend dimensions
    :return: Compiled vanilla autoencoder
    """
    encoder = (
        [
            keras.layers.Dense(
                units=hidden_layers[0], input_dim=input_dims, activation="relu"
            )
        ]
        + [
            keras.layers.Dense(units=units, activation="relu")
            for units in hidden_layers[1:]
        ]
        + [keras.layers.Dense(units=latent_dims, activation="relu")]
    )

    decoder = (
        [
            keras.layers.Dense(
                units=hidden_layers[-1],
                input_dim=latent_dims,
                activation="relu",
            )
        ]
        + [
            keras.layers.Dense(units=units, activation="relu")
            for units in hidden_layers[-2:-1:-1]
        ]
        + [keras.layers.Dense(units=input_dims, activation="sigmoid")]
    )

    encoder_model = keras.models.Sequential(encoder)
    decoder_model = keras.models.Sequential(decoder)

    auto = keras.models.Model(
        encoder_model.input,
        decoder_model(encoder_model.output),
    )

    auto.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    return encoder_model, decoder_model, auto
