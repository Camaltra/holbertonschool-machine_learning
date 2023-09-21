STOP_WORDS = {"exit", "quit", "goodbye", "bye"}
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf
import os
import numpy as np


def _question_answer(question: str, reference: str) -> str | None:
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(reference_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return None if not answer else answer


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


def question_answer(coprus_path: str) -> None:
    while True:
        in_value = input("Q:").rstrip('\n')
        if in_value.lower() in STOP_WORDS:
            print("A: Goodbye")
            break
        reference = semantic_search(coprus_path, in_value)
        answer = _question_answer(in_value, reference)
        if answer is not None:
            print(f"A: {answer}")
        else:
            print("Sorry, I do not understand your question.")
