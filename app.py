"""
app.py
------
TruthGuard AI - Flask backend.

Endpoints:
    GET  /                -> serves the frontend (templates/index.html)
    POST /check-text      -> {text: str}   -> classify with TF-IDF + Naive Bayes
    POST /check-image     -> multipart file -> OCR then classify extracted text
    POST /check-url       -> {url: str}    -> heuristic URL risk analysis

Run:
    pip install -r requirements.txt
    python train_model.py      # only needed once, to create model/*.pkl
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import re
import pickle
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
import numpy as np

from train_model import clean_text  # reuse the exact same cleaning used at train time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# ---------------------------------------------------------------------------
# Load the trained text model once at startup
# ---------------------------------------------------------------------------
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

vectorizer = None
classifier = None
if os.path.exists(VECTORIZER_PATH) and os.path.exists(CLASSIFIER_PATH):
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(CLASSIFIER_PATH, "rb") as f:
        classifier = pickle.load(f)
else:
    print("WARNING: model files not found. Run `python train_model.py` first.")


# ---------------------------------------------------------------------------
# Text classification helpers
# ---------------------------------------------------------------------------
def classify_text(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        return {"error": "Empty text provided."}

    if vectorizer is None or classifier is None:
        return {"error": "Model not loaded. Run train_model.py first."}

    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    label = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    classes = list(classifier.classes_)
    confidence = float(proba[classes.index(label)])

    top_words = top_contributing_words(cleaned, label)

    return {
        "label": label,
        "confidence": round(confidence * 100, 1),
        "top_words": top_words,
    }


def top_contributing_words(cleaned_text: str, predicted_label: str, k: int = 5):
    """
    Simple explainability: for each word in the input that the vectorizer knows,
    compare its log-probability under the predicted class vs the other class,
    and surface the words that pushed the prediction the most.
    """
    if vectorizer is None or classifier is None:
        return []

    vocab = vectorizer.vocabulary_
    classes = list(classifier.classes_)
    label_idx = classes.index(predicted_label)
    other_idx = 1 - label_idx if len(classes) == 2 else label_idx

    log_prob = classifier.feature_log_prob_  # shape (n_classes, n_features)

    words = set(cleaned_text.split())
    scored = []
    for w in words:
        if w in vocab:
            fidx = vocab[w]
            diff = log_prob[label_idx, fidx] - log_prob[other_idx, fidx]
            if diff > 0:
                scored.append((w, diff))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:k]]


# ---------------------------------------------------------------------------
# URL heuristic analysis (no external network calls required)
# ---------------------------------------------------------------------------
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "cutt.ly", "rebrand.ly",
}

SUSPICIOUS_KEYWORDS = [
    "verify", "kyc", "suspended", "blocked", "urgent", "winner", "prize",
    "lottery", "free", "gift", "claim", "reward", "click-here", "clickhere",
    "login-secure", "secure-login", "update-account", "reactivate", "bonus",
]

IP_URL_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def analyze_url(raw_url: str) -> dict:
    if not raw_url or not raw_url.strip():
        return {"error": "Empty URL provided."}

    url = raw_url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)
    host = (parsed.hostname or "").lower()

    reasons = []
    score = 0  # higher = more suspicious

    # 1. HTTPS check
    if parsed.scheme != "https":
        reasons.append("Does not use HTTPS (no valid SSL scheme detected)")
        score += 15

    # 2. IP address as host instead of a domain name
    if IP_URL_RE.match(host):
        reasons.append("Uses a raw IP address instead of a domain name")
        score += 25

    # 3. Known URL shortener
    if host in SHORTENER_DOMAINS:
        reasons.append(f"Uses a link-shortening service ({host}) which can hide the real destination")
        score += 20

    # 4. Excessive subdomains / dots
    dot_count = host.count(".")
    if dot_count >= 3:
        reasons.append("Unusually large number of subdomains in the URL")
        score += 15

    # 5. Very long URL
    if len(url) > 90:
        reasons.append("URL is unusually long")
        score += 10

    # 6. Hyphens in domain (often used to mimic brand names, e.g. "amazon-secure-login.com")
    if host.count("-") >= 2:
        reasons.append("Domain contains multiple hyphens, a common brand-spoofing pattern")
        score += 15

    # 7. Suspicious keywords anywhere in the URL
    lowered_url = url.lower()
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw.replace("-", "") in lowered_url.replace("-", "")]
    if found_keywords:
        reasons.append(f"Contains suspicious keyword(s): {', '.join(found_keywords[:4])}")
        score += 10 * min(len(found_keywords), 3)

    # 8. '@' symbol trick (browsers ignore everything before '@')
    if "@" in url:
        reasons.append("Contains '@' symbol, sometimes used to disguise the real destination")
        score += 20

    # 9. Suspicious top-level domains commonly abused for spam/phishing
    risky_tlds = (".xyz", ".top", ".click", ".work", ".gq", ".tk", ".ml", ".cf")
    if host.endswith(risky_tlds):
        reasons.append("Uses a top-level domain frequently associated with spam/phishing sites")
        score += 15

    score = min(score, 100)

    if score >= 50:
        label = "suspicious"
    elif score >= 25:
        label = "caution"
    else:
        label = "normal"

    if not reasons:
        reasons.append("No obvious red flags found in the URL structure")

    return {
        "label": label,
        "confidence": score,
        "host": host,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check-text", methods=["POST"])
def check_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = classify_text(text)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img = Image.open(file.stream).convert("L")  # grayscale improves OCR
        extracted_text = pytesseract.image_to_string(img)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    if not extracted_text.strip():
        return jsonify({
            "extracted_text": "",
            "error": "No readable text found in the image."
        }), 400

    result = classify_text(extracted_text)
    result["extracted_text"] = extracted_text.strip()
    if "error" in result and "label" not in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/check-url", methods=["POST"])
def check_url():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    result = analyze_url(url)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
