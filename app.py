from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from spacy import displacy
import re
from rouge_score import rouge_scorer

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def get_subtree_indices(token):
    indices = set()
    indices.add(token.i)
    for child in token.children:
        indices.update(get_subtree_indices(child))
    return indices

def get_root(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

def compress_sentence(sentence, level="medium"):
    doc = nlp(sentence)
    root = get_root(doc)
    if not root:
        return sentence

    # Protect only direct ROOT children that are subject/object/aux
    main_protected = {root.i}
    for child in root.children:
        if child.dep_ in {"nsubj", "nsubjpass", "dobj", "aux", "auxpass", "neg"}:
            main_protected.add(child.i)

    # Define what to remove per level
    if level == "light":
        removable_sets = [{"relcl", "appos", "acl"}]
    elif level == "medium":
        removable_sets = [
            {"relcl", "appos", "acl"},
            {"npadvmod", "advmod", "advcl"}
        ]
    else:  # heavy
        removable_sets = [
            {"relcl", "appos", "acl"},
            {"npadvmod", "advmod", "advcl"},
            {"prep", "amod", "cc", "conj", "mark"}
        ]

    remove_indices = set()

    for removable in removable_sets:
        for token in doc:
            if token.i in main_protected:
                continue
            if token.dep_ == "ROOT":
                continue
            if token.dep_ in removable:
                subtree = get_subtree_indices(token)
                safe = True
                for idx in subtree:
                    if idx in main_protected:
                        safe = False
                        break
                    if doc[idx].dep_ == "ROOT":
                        safe = False
                        break
                if safe:
                    remove_indices.update(subtree)

    return build_sentence(doc, remove_indices)

def build_sentence(doc, remove_indices):
    kept = [t for t in doc if t.i not in remove_indices]

    if len(kept) < 2:
        return " ".join([t.text for t in doc])

    result = ""
    for i, token in enumerate(kept):
        if i == 0:
            result = token.text
        else:
            if token.is_punct:
                result += token.text
            else:
                result += " " + token.text

    # Clean up spacing and punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s([.,;:!?])', r'\1', result)
    result = re.sub(r',+', ',', result)
    result = re.sub(r',\s*\.', '.', result)
    result = re.sub(r'--\s*--', '', result)
    result = re.sub(r'\s*--\s*', ' ', result)

    # Fix article grammar (an → a before consonants, a → an before vowels)
    result = re.sub(r'\ban ([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])', r'a \1', result)
    result = re.sub(r'\ba ([aeiouAEIOU])', r'an \1', result)

    result = result.strip(" ,")
    if result:
        result = result[0].upper() + result[1:]
    if result and result[-1] not in '.!?':
        result += '.'

    return result

def get_rouge_scores(original, compressed):
    scores = scorer.score(original, compressed)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 4),
        "rouge2": round(scores['rouge2'].fmeasure, 4),
        "rougeL": round(scores['rougeL'].fmeasure, 4)
    }

@app.route('/')
def home():
    return jsonify({
        "message": "Sentence Compression API is running!",
        "endpoints": {
            "compress": "POST /compress",
            "batch": "POST /batch",
            "tree": "POST /tree"
        }
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
        return jsonify({"error": "Sentence too long"}), 400
    if level not in ["light", "medium", "heavy"]:
        level = "medium"

    compressed = compress_sentence(sentence, level)
    rouge = get_rouge_scores(sentence, compressed)

    original_words = len(sentence.split())
    compressed_words = len(compressed.split())
    ratio = round(compressed_words / original_words, 2)

    return jsonify({
        "original": sentence,
        "compressed": compressed,
        "original_word_count": original_words,
        "compressed_word_count": compressed_words,
        "compression_ratio": ratio,
        "level": level,
        "rouge_scores": rouge
    })

@app.route('/batch', methods=['POST'])
def batch():
    data = request.get_json()
    if not data or 'sentences' not in data:
        return jsonify({"error": "Please provide sentences"}), 400

    sentences = data['sentences']
    level = data.get('level', 'medium')
    results = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        compressed = compress_sentence(sentence, level)
        rouge = get_rouge_scores(sentence, compressed)
        results.append({
            "original": sentence,
            "compressed": compressed,
            "original_words": len(sentence.split()),
            "compressed_words": len(compressed.split()),
            "ratio": round(len(compressed.split()) / len(sentence.split()), 2),
            "rouge1": rouge["rouge1"]
        })

    return jsonify({"results": results, "total": len(results)})

@app.route('/tree', methods=['POST'])
def tree():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"error": "Please provide a sentence"}), 400

    sentence = data['sentence'].strip()
    level = data.get('level', 'medium')
    doc = nlp(sentence)
    compressed = compress_sentence(sentence, level)
    compressed_words = set(compressed.lower().replace('.', '').split())

    options = {
        "compact": True,
        "bg": "#f8f9fa",
        "color": "#333333",
        "font": "Arial",
        "distance": 100,
        "arrow_stroke": 2,
        "arrow_width": 8
    }

    svg = displacy.render(doc, style="dep", options=options, page=False)
    removed_words = [t.text for t in doc if t.text.lower() not in compressed_words and not t.is_punct]

    return jsonify({
        "svg": svg,
        "removed_words": removed_words,
        "compressed": compressed
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)