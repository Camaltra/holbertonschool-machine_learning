#!/usr/bin/env python3

"""useless comments"""


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Compute the forward propagation of the NN
    :param x: The input layer
    :param layer_sizes: A list contain number of node for
                        each layers
    :param activations: A list contain the activation
                        function for each layer
    :return: The final tensor
    """
    predictions = x
    for layer_size, activation in zip(layer_sizes, activations):
        predictions = create_layer(predictions, layer_size, activation)
    return predictions
