# TruthGuard AI

A fake news / spam / phishing detector with three input modes: **Text**, **Image
(screenshot OCR)**, and **URL**. Flask backend + a plain HTML/CSS/JS frontend.

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Tesseract OCR (system package, not pip)
#    macOS:   brew install tesseract
#    Ubuntu:  sudo apt install tesseract-ocr
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki
#    (If tesseract isn't on your PATH, set the path explicitly at the top
#     of app.py: pytesseract.pytesseract.tesseract_cmd = r"C:\...\tesseract.exe")

# 3. Train the demo text-classification model (creates model/*.pkl)
python train_model.py

# 4. Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works

| Mode  | Pipeline |
|-------|----------|
| Text  | clean text → TF-IDF vectorizer → Multinomial Naive Bayes → label + confidence + top contributing words |
| Image | image → Tesseract OCR → extracted text → same text pipeline as above |
| URL   | parsed URL → heuristic checks (HTTPS, IP-as-host, shorteners, hyphens, risky TLDs, suspicious keywords, `@` trick, length) → risk score |

## ⚠️ About accuracy — read this before treating results as real

`train_model.py` currently trains on a **small, hand-written demo dataset**
(~60 examples). It's enough to prove the pipeline works, but **not enough for
real-world accuracy**. Before using this for anything beyond a demo/project
submission:

1. Replace `load_dataset()` in `train_model.py` with a real labelled dataset —
   e.g. a spam/ham CSV (thousands of rows) plus a fake-news dataset, merged
   into the same `["suspicious", "normal"]` label scheme.
2. Re-run `python train_model.py` to retrain and check the printed
   precision/recall report.
3. Consider swapping Naive Bayes for Logistic Regression or a small
   transformer model once you have more data — the vectorizer/classifier
   interface in `app.py` stays the same either way.

The URL checker is fully rule-based (no external API calls), so it works
offline but won't catch a brand-new phishing domain with a clean-looking URL.
For stronger URL checks later, you could add a WHOIS domain-age lookup or a
call to a threat-intelligence API (e.g. Google Safe Browsing).

## Project structure

```
truthguard-ai/
├── app.py              # Flask backend (routes + OCR + URL heuristics)
├── train_model.py       # Builds demo dataset + trains TF-IDF/Naive Bayes
├── requirements.txt
├── model/                # vectorizer.pkl, classifier.pkl (generated)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js
```
