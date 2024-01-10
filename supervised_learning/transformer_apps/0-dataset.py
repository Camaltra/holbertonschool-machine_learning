#!/usr/bin/python3

"""
Useless
"""

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Class that loads and preps a dataset for machine translation
    """

    def __init__(self):
        """
        Init the class
        """

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Instance method that creates sub-word tokenizers for our dataset
        :param data: a tf.data.Dataset
        :return: tokenizer_pt, tokenizer_en
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2**15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
