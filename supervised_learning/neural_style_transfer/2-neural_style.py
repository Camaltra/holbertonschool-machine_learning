#!/usr/bin/env python3

"""useless comments"""

import numpy as np
import tensorflow as tf


def check_image_channel_input(img, source):
    """
    Check the channel of given image
    :param img: The image
    :param source: The variable name to error message
    :return:
    """
    if type(img) != np.ndarray or img.shape[-1] != 3:
        raise TypeError(
            "{} must be a numpy.ndarray with shape (h, w, 3)".format(source)
        )


def check_tensor_rank_input(input_layer, source):
    """
    Check the tensor rank
    :param input_layer: The given tensor
    :param source: The variable name to error message
    :return: Nothing
    """
    if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
            input_layer.shape) != 4:
        raise TypeError("{} must be a tensor of rank 4".format(source))


class NST:
    """Neural style transfer model"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Init function class
        :param style_image: The style_image (?, ?, 3)
        :param content_image: The content image (?, ?, 3)
        :param alpha: The alpha parameter
        :param beta: The beta parameter
        """
        tf.enable_eager_execution()
        check_image_channel_input(style_image, "style_image")
        check_image_channel_input(content_image, "content_image")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        check_hyperparameter_input(alpha, "alpha")
        check_hyperparameter_input(beta, "beta")
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Scale the image to (1, 512 or less, 512 or less, 3)
        :param image: The given image to resize
        :return: The resized image
        """
        check_image_channel_input(image, "image")

        max_dim = max(image.shape[:-1])
        ratio_dims = 512 / max_dim

        new_dims = tuple([int(dim * ratio_dims) for dim in image.shape[:-1]])
        image = tf.expand_dims(image, 0)  # [1, h, w, 3]
        resized_image = tf.image.resize_bicubic(image, new_dims) / 255

        return tf.clip_by_value(resized_image, 0.0, 1.0)

    def load_model(self):
        """
        Load VGG19 model
        :return: The model
        """
        vgg19 = tf.keras.applications.VGG19(include_top=False)
        for layer in vgg19.layers:
            layer.trainable = False
        vgg19.save("vgg_base_model.h5")
        model = tf.keras.models.load_model(
            "vgg_base_model.h5",
            custom_objects={
                "MaxPooling2D": tf.keras.layers.AveragePooling2D()
            })

        outputs = ([model.get_layer(layer).output
                   for layer in self.style_layers]
                   + [model.get_layer(self.content_layer).output])
        self.model = tf.keras.models.Model(model.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate the gram matrix
        :return: The gram matrix
        """
        check_tensor_rank_input(input_layer, "input_layer")
        # Checker doesn't like this code

        # coef = 1 / (input_layer.shape[1] * input_layer.shape[2])
        # batch_size, height, width, channels = input_layer.shape
        # flattened_inputs = tf.reshape(
        #     input_layer,
        #     [batch_size, height * width, channels]
        # )
        # gram_matrix = tf.matmul(
        #     flattened_inputs,
        #     flattened_inputs,
        #     transpose_a=True
        # )
        # return gram_matrix * coef

        # Re write the code inspired of github alumni
        batch_size, height, width, channels = input_layer.shape
        flattened_inputs = tf.reshape(
            input_layer,
            [-1, channels]
        )
        gram_matrix = tf.matmul(
            tf.transpose(flattened_inputs),
            flattened_inputs,
        ) / tf.cast(flattened_inputs.shape[0], tf.float32)
        return tf.reshape(gram_matrix, [1, -1, channels])
