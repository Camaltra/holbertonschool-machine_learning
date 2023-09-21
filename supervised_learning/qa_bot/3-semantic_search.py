import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") is False:
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            documents.append(f.read())
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(module_url)
    embed = model(documents)
    corr = np.inner(embed, embed)
    close = np.argmax(corr[0, 1:])
    similarity = documents[close + 1]
    return similarity