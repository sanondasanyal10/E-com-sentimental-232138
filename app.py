import os
import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import emoji

# -------------------------------
# Config
# -------------------------------
APP_NAME = "ReviewSentiment"
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # allow utf-8 in JSON

# -------------------------------
# Text normalization
# -------------------------------
_bn_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.translate(_bn_digits)
    s = s.lower()
    s = emoji.demojize(s, language="en")
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^\w\s:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------------
# Load Model
# -------------------------------
LABEL_MAP = {0: "Negative", 1: "Positive"}  # adjust as needed
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"[WARN] Model file not found at {MODEL_PATH}")

# -------------------------------
# Prediction Helper
# -------------------------------
def predict_texts(texts):
    if model is None:
        raise RuntimeError("Model not loaded. Please upload best_model.pkl.")

    norm = [normalize_text(t) for t in texts]
    try:
        probs = model.predict_proba(norm)[:, 1]
    except Exception:
        scores = model.decision_function(norm)
        probs = 1 / (1 + np.exp(-scores))

    preds = (probs >= 0.65).astype(int)
    return [
        {
            "text": raw,
            "label_id": int(p),
            "label": LABEL_MAP.get(int(p), str(int(p))),
            "probability": float(pr)
        }
        for raw, p, pr in zip(texts, preds, probs)
    ]

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", app_name=APP_NAME)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
            texts = data.get("texts")
            if not texts or not isinstance(texts, list):
                return jsonify({"error": "Provide JSON with key 'texts' as a non-empty list."}), 400
        else:
            text = request.form.get("text", "").strip()
            if not text:
                return jsonify({"error": "Missing 'text' in form data."}), 400
            texts = [text]

        results = predict_texts(texts)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
