#!/usr/bin/env python3


"""useless comment"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    iterations=5,
    seed=0,
    workers=1,
):
    """Useless function"""
    model = Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        iter=iterations,
        seed=seed,
        workers=workers,
    )
    model.train(
        sentences,
        epochs=iterations,
        total_examples=model.corpus_count
    )
    return model
