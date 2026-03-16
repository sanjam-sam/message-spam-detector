from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
MODEL_PATH = BASE_DIR / "spam_model.pkl"

app = Flask(__name__)

vectorizer = None
model = None


def load_artifacts():
    global vectorizer, model
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run spam_detector.py once to create vectorizer.pkl and spam_model.pkl."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "spam_detector.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()

    if not subject:
        return jsonify({"error": "Subject is required."}), 400

    features = vectorizer.transform([subject])
    prediction = model.predict(features)[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
