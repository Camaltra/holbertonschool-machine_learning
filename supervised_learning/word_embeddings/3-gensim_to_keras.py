#!/usr/bin/env python3


"""useless comment"""


def gensim_to_keras(model):
    """
    From a gensim model, traform it to keras embeding layer
    :param model: The gensim model
    :return: The keras layer
    """
    return model.wv.get_keras_embedding(train_embeddings=False)