#!/usr/bin/env python3


"""useless comment"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Build sparse autoencoder
    :param input_dims: Input dimensions
    :param hidden_layers: Hidden layers unit values
    :param latent_dims: Latend dimensions
    :param lambtha: The rate for the l2 regularizer
    :return: Compiled sparse autoencoder
    """
    encoder_input = keras.layers.Input(
        shape=(input_dims,),
        name="encoder_input",
    )
    encoder_layer = encoder_input
    for units in hidden_layers:
        encoder_layer = keras.layers.Dense(units=units, activation="relu")(
            encoder_layer
        )
    encoder_output = keras.layers.Dense(
        units=latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha),
        name="encoder_latent",
    )(encoder_layer)

    encoder_model = keras.models.Model(
        encoder_input,
        encoder_output,
        name="encoder",
    )

    decoder_input = keras.layers.Input(
        shape=(latent_dims,),
        name="decoder_input",
    )
    decoder_layer = decoder_input
    for units in reversed(hidden_layers):
        decoder_layer = keras.layers.Dense(units=units, activation="relu")(
            decoder_layer
        )
    decoder_output = keras.layers.Dense(
        units=input_dims,
        activation="sigmoid",
    )(decoder_layer)

    decoder_model = keras.models.Model(
        decoder_input,
        decoder_output,
        name="decoder",
    )

    auto = keras.models.Model(
        encoder_model.input,
        decoder_model(encoder_model(encoder_model.input)),
    )

    auto.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    return encoder_model, decoder_model, auto
