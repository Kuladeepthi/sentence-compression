from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import re

app = Flask(__name__)
CORS(app)

# Load model once at startup
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

@app.route('/')
def home():
    return jsonify({
        "message": "Sentence Compression API is running!",
        "usage": "POST /compress with JSON body: {sentence: 'your sentence here'}"
    })

@app.route('/compress', methods=['POST'])
def compress():
    data = request.get_json()

    if not data or 'sentence' not in data:
        return jsonify({"error": "Please provide a sentence"}), 400

    sentence = data['sentence'].strip()

    if len(sentence) == 0:
        return jsonify({"error": "Sentence cannot be empty"}), 400

    if len(sentence) > 1000:
        return jsonify({"error": "Sentence too long, max 1000 characters"}), 400

    compressed = compress_sentence(sentence)

    original_words = len(sentence.split())
    compressed_words = len(compressed.split())
    ratio = round(compressed_words / original_words, 2)

    return jsonify({
        "original": sentence,
        "compressed": compressed,
        "original_word_count": original_words,
        "compressed_word_count": compressed_words,
        "compression_ratio": ratio
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)