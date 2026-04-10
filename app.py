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

def compress_sentence(sentence, level="medium", target_words=None):
    doc = nlp(sentence)
    
    # Priority order of what to remove (from safest to most aggressive)
    removal_priority = [
        {"relcl", "appos", "acl"},                          # Level 1 - safest
        {"npadvmod", "advmod", "advcl"},                     # Level 2
        {"prep", "mark", "cc", "conj"},                      # Level 3
        {"amod", "compound", "det", "quantmod", "nummod"},   # Level 4 - most aggressive
    ]

    if level == "light":
        removable_sets = removal_priority[:1]
    elif level == "medium":
        removable_sets = removal_priority[:2]
    else:
        removable_sets = removal_priority[:3]

    # If target_words specified, remove progressively
    if target_words:
        removable_sets = removal_priority

    protected = {"ROOT", "nsubj", "nsubjpass", "dobj", "aux", "auxpass", "neg"}
    remove_indices = set()

    for removable in removable_sets:
        for token in doc:
            if token.dep_ in protected:
                continue
            if token.dep_ in removable:
                subtree = get_subtree_indices(token)
                safe = True
                for idx in subtree:
                    if doc[idx].dep_ in protected:
                        safe = False
                        break
                if safe:
                    remove_indices.update(subtree)

        # Check if target reached
        if target_words:
            kept = [t for t in doc if t.i not in remove_indices and not t.is_punct]
            if len(kept) <= target_words:
                break

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

    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s([.,;:!?])', r'\1', result)
    result = re.sub(r',+', ',', result)
    result = re.sub(r',\s*\.', '.', result)
    result = re.sub(r'--\s*--', '', result)
    result = re.sub(r'\s*--\s*', ' ', result)
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

def get_dependency_tree(sentence, remove_indices_words):
    doc = nlp(sentence)
    remove_set = set(remove_indices_words)

    # Build custom SVG using displacy data
    words = []
    arcs = []

    for token in doc:
        color = "#ff6b6b" if token.text in remove_set else "#51cf66"
        words.append({
            "text": token.text,
            "tag": token.dep_,
            "color": color
        })

    for token in doc:
        if token.dep_ != "ROOT":
            start = min(token.i, token.head.i)
            end = max(token.i, token.head.i)
            direction = "left" if token.i < token.head.i else "right"
            arcs.append({
                "start": start,
                "end": end,
                "label": token.dep_,
                "dir": direction
            })

    options = {
        "compact": True,
        "bg": "#ffffff",
        "color": "#333333",
        "font": "Arial"
    }

    svg = displacy.render(doc, style="dep", options=options, page=False)
    return svg

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
    target_words = data.get('target_words', None)

    if len(sentence) == 0:
        return jsonify({"error": "Sentence cannot be empty"}), 400
    if len(sentence) > 1000:
        return jsonify({"error": "Sentence too long"}), 400
    if level not in ["light", "medium", "heavy"]:
        level = "medium"

    if target_words:
        try:
            target_words = int(target_words)
        except:
            target_words = None

    compressed = compress_sentence(sentence, level, target_words)
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
        return jsonify({"error": "Please provide sentences array"}), 400

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

    removed_words = []
    for token in doc:
        if token.text.lower() not in compressed_words and not token.is_punct:
            removed_words.append(token.text)

    options = {
        "compact": False,
        "bg": "#f8f9fa",
        "color": "#333333",
        "font": "Arial",
        "distance": 120,
        "arrow_stroke": 2,
        "arrow_width": 8
    }

    svg = displacy.render(doc, style="dep", options=options, page=False)
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