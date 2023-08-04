#!/usr/bin/env python3


"""useless comment"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Build convolutinnal autoencoder
    :param input_dims: Input dimensions
    :param hidden_layers: Hidden layers unit values
    :param latent_dims: Latend dimensions
    :return: Compiled convolutinnal autoencoder
    """

    encoder_input = keras.layers.Input(shape=input_dims)
    encoder_layer = encoder_input
    for encoder_filter in filters:
        encoder_layer = keras.layers.Conv2D(
            filters=encoder_filter,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )(encoder_layer)
        encoder_layer = keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=None, padding="same"
        )(encoder_layer)

    encoder_model = keras.models.Model(
        encoder_input,
        encoder_layer,
        name="encoder",
    )

    decoder_input = keras.layers.Input(shape=latent_dims)
    decoder_layer = decoder_input
    for decoder_filter in reversed(filters[1:]):
        decoder_layer = keras.layers.Conv2D(
            filters=decoder_filter,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )(decoder_layer)
        decoder_layer = keras.layers.UpSampling2D(size=(2, 2))(decoder_layer)
    decoder_layer = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
    )(decoder_layer)
    decoder_layer = keras.layers.UpSampling2D(size=(2, 2))(decoder_layer)
    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
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
