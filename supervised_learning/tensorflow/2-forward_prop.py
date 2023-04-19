#!/usr/bin/env python3

"""useless comments"""


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    predictions = x
    for layer_size, activation in zip(layer_sizes, activations):
        predictions = create_layer(predictions, layer_size, activation)
    return predictions
