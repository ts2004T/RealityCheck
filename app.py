from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

model = None
vectorizer = None

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ----------------------------
# Utility functions
# ----------------------------
def load_resources():
    global model, vectorizer
    if model is None or vectorizer is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            raise FileNotFoundError("Model or vectorizer not found.")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(tokens)

# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "RealityCheck backend is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_resources()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    clean_text = preprocess_text(text)
    features = vectorizer.transform([clean_text])
    result = model.predict(features)[0]

    return jsonify({
        "prediction": "REAL NEWS" if result == 1 else "FAKE NEWS"
    })

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
