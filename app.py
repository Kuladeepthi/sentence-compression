from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import re

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

def get_subtree_indices(token):
    """Get all indices in a token's subtree"""
    indices = set()
    indices.add(token.i)
    for child in token.children:
        indices.update(get_subtree_indices(child))
    return indices

def is_safe_to_remove(token, doc):
    """Check if removing this token's subtree is safe"""
    # Never remove ROOT
    if token.dep_ == "ROOT":
        return False
    # Never remove main subject
    if token.dep_ in {"nsubj", "nsubjpass"}:
        return False
    # Never remove main object
    if token.dep_ == "dobj":
        return False
    # Never remove auxiliary verbs of ROOT
    if token.dep_ in {"aux", "auxpass"} and token.head.dep_ == "ROOT":
        return False
    # Never remove negation
    if token.dep_ == "neg":
        return False
    return True

def compress_sentence(sentence, level="medium"):
    doc = nlp(sentence)
    remove_indices = set()

    # Define what to remove per level
    if level == "light":
        # Only remove clearly optional clauses
        removable = {"relcl", "appos"}
    elif level == "medium":
        # Remove optional clauses + adverbial modifiers
        removable = {"relcl", "appos", "advcl", "advmod", "npadvmod", "acl"}
    else:  # heavy
        # Remove everything optional including prepositional phrases
        removable = {"relcl", "appos", "advcl", "advmod", "npadvmod",
                    "acl", "prep", "amod", "cc", "conj", "mark", "quantmod"}

    for token in doc:
        if token.dep_ in removable and is_safe_to_remove(token, doc):
            subtree = get_subtree_indices(token)
            # Extra safety - make sure we are not removing subject or object
            safe = True
            for idx in subtree:
                t = doc[idx]
                if t.dep_ in {"nsubj", "nsubjpass", "dobj", "ROOT"}:
                    safe = False
                    break
            if safe:
                remove_indices.update(subtree)

    # Build compressed sentence keeping order
    kept_tokens = [token for token in doc if token.i not in remove_indices]

    if len(kept_tokens) < 2:
        return sentence  # Safety: return original if too much removed

    # Reconstruct with proper spacing
    compressed = ""
    for i, token in enumerate(kept_tokens):
        if i == 0:
            compressed = token.text
        else:
            # Add space before token unless it's punctuation
            if token.is_punct:
                compressed += token.text
            else:
                compressed += " " + token.text

    # Clean up
    compressed = re.sub(r'\s+', ' ', compressed)
    compressed = re.sub(r'\s([.,;:!?])', r'\1', compressed)
    compressed = re.sub(r',+', ',', compressed)
    compressed = re.sub(r',\s*\.', '.', compressed)
    compressed = re.sub(r'--\s*--', '', compressed)
    compressed = re.sub(r'\s*--\s*', ' ', compressed)
    compressed = compressed.strip(" ,")

    # Capitalize first letter
    if compressed:
        compressed = compressed[0].upper() + compressed[1:]

    # End with period
    if compressed and compressed[-1] not in '.!?':
        compressed += '.'

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