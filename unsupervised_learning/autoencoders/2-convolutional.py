#!/usr/bin/env python3


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Build convolutinnal autoencoder
    :param input_dims: Input dimensions
    :param hidden_layers: Hidden layers unit values
    :param latent_dims: Latend dimensions
    :return: Compiled convolutinnal autoencoder
    """
    encoder = [
        keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=input_dims,
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
    ]

    for num_filters in filters[1:]:
        encoder += [
            keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        ]

    decoder = [
        keras.layers.Conv2D(
            filters=filters[-1],
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=latent_dims,
        ),
        keras.layers.UpSampling2D(size=(2, 2)),
    ]

    for num_filters in filters[-2:0:-1]:
        decoder += [
            keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            keras.layers.UpSampling2D(size=(2, 2)),
        ]

    decoder += [
        keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding="valid",
            activation="relu",
        ),
        keras.layers.UpSampling2D(size=(2, 2)),
        keras.layers.Conv2D(
            filters=input_dims[-1],
            kernel_size=(3, 3),
            padding="same",
            activation="sigmoid",
        ),
    ]

    encoder_model = keras.models.Sequential(encoder)
    decoder_model = keras.models.Sequential(decoder)

    print(encoder_model.summary())
    print(decoder_model.summary())

    auto = keras.models.Model(
        encoder_model.input,
        decoder_model(encoder_model.output),
    )

    auto.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    return encoder_model, decoder_model, auto
