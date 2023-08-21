#!/usr/bin/env python3


"""useless comment"""

import re
import numpy as np


WORD_DELIMITER = r"""[,.!?#$%&'*+,-./:;<>=@\^_`|~"(){} ]+"""


class CountVectorizer:
    def __init__(self, vocab=None):
        """
        Init
        :param vocab: The prior vocab if one
        """
        self.vocab = vocab
        self.processed_formatted_vocab = None
        self.features = []

    @staticmethod
    def check_and_format_word(word):
        """
        Check if a word is correct, and format it
        :param word: The word to check
        Return: None is not a right word, else word.lower()
        """
        if len(word) < 2:
            return None
        return word.lower()

    @staticmethod
    def format_vocab(vocab, sort_values):
        """
        Fomart the vocab in a dict {word: idx}
        :param vocab: The vocab
        :return: The formatted vocab
        """
        if sort_values:
            return {word: idx for idx, word in enumerate(sorted(vocab))}
        return {word: idx for idx, word in enumerate(vocab)}

    def process_sentence(self, sentence):
        """
        Process a sentence to extract all words
        :param sentence: A sentence
        :return: A list of word that occur in the sentence
        """
        processed_sentence = []
        splitted_sentence = re.split(WORD_DELIMITER, sentence)
        for word in splitted_sentence:
            formatted_word = self.check_and_format_word(word)
            if formatted_word is not None:
                processed_sentence.append(formatted_word)

        return processed_sentence

    def create_vocab(self, sentences):
        """
        Get a list of all word that appear in given sentences
        :param sentences: Sentences
        :return: A list of all word that appear in the given sentences
        """
        vocab = set()
        for sentence in sentences:
            sub_vocab = set(self.process_sentence(sentence))
            vocab = vocab.union(sub_vocab)
        return list(vocab)

    def fit(self, X, y=None):
        """
        Fit the class to the data
        :param X: The dataset (ie Sentences)
        :param y: The label if exist
        :return: Self
        """
        sort_values = False
        if self.vocab is None:
            sort_values = True
            self.vocab = self.create_vocab(X)
        self.processed_formatted_vocab = self.format_vocab(self.vocab, sort_values)
        self.features = list(self.processed_formatted_vocab.keys())
        return self

    def transform(self, X):
        """
        Transform given data through prior fit
        :param X: The data
        :return: The transformed data
        """
        bag_of_word = np.zeros((len(X), len(self.vocab)), dtype=np.int)
        for idx, sentence in enumerate(X):
            processed_sentence = self.process_sentence(sentence)
            for word in processed_sentence:
                if word in self.processed_formatted_vocab.keys():
                    word_idx = self.processed_formatted_vocab.get(word)
                    bag_of_word[idx][word_idx] += 1

        return bag_of_word

    def fit_transform(self, X, y=None):
        """
        Process fit & transform operation
        :param X: The dataset
        :param y: The label if needed
        :return: self.transform()
        """
        self.fit(X, y)
        return self.transform(X)


def bag_of_words(sentences, vocab=None):
    """
    Create a bag of word
    :param sentences: A list of sentences
    :param vocab: The vocab is needed
    :return: The bag, the vocab
    """
    count_vectorizer = CountVectorizer(vocab)
    bag_of_word = count_vectorizer.fit_transform(sentences)
    return bag_of_word, count_vectorizer.features
