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


def check_hyperparameter_input(hyperparameter, source):
    """
    Check given hyperparameter
    :param hyperparameter: The hyperparameter
    :param source: The variable name to error message
    :return:
    """
    if type(hyperparameter) not in [float, int] or hyperparameter < 0:
        raise TypeError("{} must be a non-negative number".format(source))


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
        self.generate_features()

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

    def generate_features(self):
        """
        Forward propagation of our 2 images throught the model
        Saved the content and style feature representations from our model
        :return: Nothing
        """
        preprocess_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        preprocess_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        style_outputs = style_output[:-1]
        content_ouput = content_output[-1]

        self.gram_style_features = [self.gram_matrix(layer)
                                    for layer in
                                    style_outputs]
        self.content_feature = content_ouput

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        :param style_output: The style output from a layer
        :param gram_target: The targat value
        :return: The layers cost
        """
        check_tensor_rank_input(style_output, "style_output")
        output_channel = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != [1, output_channel, output_channel]:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    output_channel,
                    output_channel
                )
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Get the style cost
        :param style_outputs: Style output for the generated image
        :return: The total cost for style
        """
        if not isinstance(style_outputs, list) or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)
                )
            )
        total_cost = 0.0
        weight = 1.0 / float(len(style_outputs))
        for style, target in zip(style_outputs, self.gram_style_features):
            total_cost += weight * self.layer_style_cost(
                style, target
            )
        return total_cost

    def content_cost(self, content_output):
        """
        Compute the content loss
        :param content_output: A tf.Tensor containing the content
                               output for the generated image
        :return: The content cost
        """
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError("content_output must be a tensor of shape {}"
                            .format(self.content_feature.shape))
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Compute the total cost
        :param generated_image: The generated image
        :return: The total cost, content cost, style cost
        """
        shape_content = self.content_image.shape
        if (not isinstance(generated_image, (tf.Variable, tf.Tensor)) or
                generated_image.shape != shape_content):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content))

        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)

        generated_output = self.model(generated_image)

        generated_content = generated_output[-1]
        generated_style = generated_output[:-1]
        content_cost = self.content_cost(generated_content)
        style_cost = self.style_cost(generated_style)

        return (self.alpha * content_cost + self.beta * style_cost,
                content_cost,
                style_cost)

    def compute_grads(self, generated_image):
        """
        Compute the grads for a certain step
        :param generated_image: The generated image
        :return: Return the computed gradient
        """
        with tf.GradientTape() as tape:
            all_loss = self.total_cost(generated_image)
        total_loss, content_loss, style_loss = all_loss
        return (tape.gradient(total_loss, generated_image), total_loss,
                content_loss, style_loss)
