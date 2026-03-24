
import spacy
import re

nlp = spacy.load("en_core_web_sm")

REMOVE_DEPS = {"relcl", "prep", "advmod", "npadvmod", "advcl", "appos", "pobj", "amod"}

def compress_sentence(sentence):
    doc = nlp(sentence)
    remove_indices = set()

    def mark_subtree(token):
        remove_indices.add(token.i)
        for child in token.children:
            mark_subtree(child)

    for token in doc:
        if token.dep_ in REMOVE_DEPS:
            mark_subtree(token)

    compressed = " ".join([token.text for token in doc if token.i not in remove_indices])

    compressed = re.sub(r'\s+', ' ', compressed)
    compressed = re.sub(r'\s([.,;:!?])', r'\1', compressed)
    compressed = re.sub(r',+', ',', compressed)
    compressed = re.sub(r',\s*\.', '.', compressed)
    compressed = re.sub(r'\(\s*\)', '', compressed)
    compressed = re.sub(r'--\s*--', '', compressed)
    compressed = compressed.strip(" ,")
    if not compressed.endswith('.'):
        compressed += '.'

    return compressed
