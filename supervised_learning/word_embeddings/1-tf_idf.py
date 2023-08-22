#!/usr/bin/env python3


"""useless comment"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create the matrix of TF - IDF
    :param sentences: A list of sentences
    :param vocab: The vocab is needed
    :return: The matrix, the vocab
    """
    tf_idf_vectorizer = TfidfVectorizer(vocabulary=vocab)
    output = tf_idf_vectorizer.fit_transform(sentences)

    return output.toarray(), tf_idf_vectorizer.get_feature_names()
