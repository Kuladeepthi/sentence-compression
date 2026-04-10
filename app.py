from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import re

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

# Carefully tuned compression levels
# light  - only removes extra descriptive clauses
# medium - removes clauses + time/place adverbials
# heavy  - removes everything except core subject+verb+object
COMPRESSION_LEVELS = {
    "light": {
        "relcl",      # relative clause (who graduated from Harvard)
        "appos",      # apposition (John, the CEO, ...)
        "acl",        # clausal modifier
    },
    "medium": {
        "relcl",      # relative clause
        "appos",      # apposition
        "acl",        # clausal modifier
        "advcl",      # adverbial clause (because, although...)
        "npadvmod",   # noun phrase adverbial (last year)
        "advmod",     # adverb modifier (recently, quickly)
    },
    "heavy": {
        "relcl",      # relative clause
        "appos",      # apposition
        "acl",        # clausal modifier
        "advcl",      # adverbial clause
        "npadvmod",   # noun phrase adverbial
        "advmod",     # adverb modifier
        "prep",       # prepositional phrase (on climate change)
        "pobj",       # object of preposition
        "amod",       # adjectival modifier (young, prestigious)
        "det",        # determiner (the, a)
        "cc",         # coordinating conjunction (and, but)
        "conj",       # conjunct
        "mark",       # marker (that, which)
        "quantmod",   # quantifier modifier
        "nummod",     # numeric modifier
    }
}

def compress_sentence(sentence, level="medium"):
    doc = nlp(sentence)
    remove_indices = set()
    REMOVE_DEPS = COMPRESSION_LEVELS.get(level, COMPRESSION_LEVELS["medium"])

    def mark_subtree(token):
        remove_indices.add(token.i)
        for child in token.children:
            mark_subtree(child)

    for token in doc:
        if token.dep_ in REMOVE_DEPS:
            # Never remove ROOT token (main verb)
            if token.dep_ == "ROOT":
                continue
            # Never remove subject (nsubj, nsubjpass)
            if token.dep_ in {"nsubj", "nsubjpass"}:
                continue
            # Never remove direct object (dobj)
            if token.dep_ == "dobj":
                continue
            mark_subtree(token)

    # Always keep ROOT, nsubj, dobj tokens
    for token in doc:
        if token.dep_ in {"ROOT", "nsubj", "nsubjpass", "dobj"}:
            remove_indices.discard(token.i)

    compressed = " ".join([token.text for token in doc if token.i not in remove_indices])

    # Clean up punctuation and spacing
    compressed = re.sub(r'\s+', ' ', compressed)
    compressed = re.sub(r'\s([.,;:!?])', r'\1', compressed)
    compressed = re.sub(r',+', ',', compressed)
    compressed = re.sub(r',\s*\.', '.', compressed)
    compressed = re.sub(r'\(\s*\)', '', compressed)
    compressed = re.sub(r'--\s*--', '', compressed)
    compressed = re.sub(r'\s*-\s*-\s*', ' ', compressed)
    compressed = compressed.strip(" ,")

    # Make sure it ends with a period
    if compressed and not compressed[-1] in '.!?':
        compressed += '.'

    # If compression removed too much (less than 2 words), return original
    if len(compressed.split()) < 2:
        return sentence

    return compressed

@app.route('/')
def home():
    return jsonify({
        "message": "Sentence Compression API is running!",
        "usage": "POST /compress with JSON body: {sentence: 'your sentence', level: 'light/medium/heavy'}"
    })

@app.route('/compress', methods=['POST'])
def compress():
    data = request.get_json()

    if not data or 'sentence' not in data:
        return jsonify({"error": "Please provide a sentence"}), 400

    sentence = data['sentence'].strip()
    level = data.get('level', 'medium')

    if len(sentence) == 0:
        return jsonify({"error": "Sentence cannot be empty"}), 400

    if len(sentence) > 1000:
        return jsonify({"error": "Sentence too long, max 1000 characters"}), 400

    if level not in ["light", "medium", "heavy"]:
        level = "medium"

    compressed = compress_sentence(sentence, level)

    original_words = len(sentence.split())
    compressed_words = len(compressed.split())
    ratio = round(compressed_words / original_words, 2)

    return jsonify({
        "original": sentence,
        "compressed": compressed,
        "original_word_count": original_words,
        "compressed_word_count": compressed_words,
        "compression_ratio": ratio,
        "level": level
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)