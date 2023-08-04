#!/usr/bin/env python3


"""useless comment"""


import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Sampling layer class"""

    def call(self, inputs, *arg, **kwargs):
        """
        Call of the sampling layer
        :param inputs: Inputs
        :return: A layer
        """
        z_mean, z_log_var = inputs
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


class VAE:
    """Class for the variationnal auto-encodeur"""

    def __init__(self, input_dims, hidden_layers, latent_dims):
        """
        Init the VAE model
        :param input_dims: Input dimensions
        :param hidden_layers: Hidden layers unit values
        :param latent_dims: Latend dimensions
        """
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.encoder = self._build_encoder(hidden_layers)
        self.decoder = self._build_decoder(hidden_layers)
        self.model = self._build_model()
        self.z_mean = None
        self.z_log_var = None

    def _build_encoder(self, hidden_layers):
        """
        Build the encoder
        :param hidden_layers: Hidden layers unit values
        :return: The encoder model
        """
        encoder_input = keras.layers.Input(shape=(self.input_dims,))
        layer = encoder_input
        for units in hidden_layers:
            layer = keras.layers.Dense(units=units, activation="relu")(layer)
        z_mean = keras.layers.Dense(
            units=self.latent_dims, activation="linear", name="z_mean"
        )(layer)
        z_log_var = keras.layers.Dense(
            units=self.latent_dims, activation="linear", name="z_log_var"
        )(layer)
        z = Sampling()([z_mean, z_log_var])
        return keras.models.Model(
            encoder_input, [z_mean, z_log_var, z], name="encoder"
        )

    def _build_decoder(self, hidden_layers):
        """
        Build the decoder model
        :param hidden_layers: Hidden layers unit values
        :return: The decoder model
        """
        decoder_input = keras.layers.Input(shape=(self.latent_dims,))
        layer = decoder_input
        for units in reversed(hidden_layers):
            layer = keras.layers.Dense(units=units, activation="relu")(layer)
        output = keras.layers.Dense(
            units=self.input_dims, activation="sigmoid", name="output"
        )(layer)

        return keras.models.Model(decoder_input, output, name="decoder")

    def _build_model(self):
        """
        Build the auto-encoder model
        :return: The compiled auto-encoder model
        """
        encoder_input = self.encoder.inputs
        decoder_output = self.decoder(self.encoder(encoder_input)[2])
        auto_encoder = keras.models.Model(encoder_input, decoder_output)

        def loss(x, x_decoded):
            mse = keras.backend.sum(
                keras.backend.binary_crossentropy(x, x_decoded), axis=1
            )
            z_mean, z_log_var, _ = self.encoder(x)
            kl_loss = -0.5 * keras.backend.sum(
                1
                + z_log_var
                - keras.backend.square(z_mean)
                - keras.backend.exp(z_log_var),
                axis=-1,
            )
            return keras.backend.mean(mse + kl_loss)

        auto_encoder.compile(optimizer=keras.optimizers.Adam(), loss=loss)

        return auto_encoder


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build variatinnal autoencoder
    :param input_dims: Input dimensions
    :param hidden_layers: Hidden layers unit values
    :param latent_dims: Latend dimensions
    :return: The encoder, decoder and the variatinnal auto-encoder
    """
    auto_encoder = VAE(input_dims, hidden_layers, latent_dims)
    return auto_encoder.encoder, auto_encoder.decoder, auto_encoder.model
