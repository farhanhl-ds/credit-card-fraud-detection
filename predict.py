"""Flask web service for the fraud model.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> fraud probability and decision for one transaction

Run locally:   gunicorn --bind 0.0.0.0:9696 predict:app
Run in dev:    python predict.py
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

artifact = joblib.load(MODEL_PATH)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
numeric_features = set(artifact["numeric_features"])

app = Flask("fraud-service")


def to_frame(payload):
    """One-row DataFrame in the training column order with the training dtypes.

    Unknown keys are ignored. Missing features become NaN and are imputed by
    the preprocessing pipeline; they are reported back to the caller.
    """
    row = pd.DataFrame([payload]).reindex(columns=features)
    for col in features:
        if col in numeric_features:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        else:
            row[col] = row[col].astype(object)
    return row


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with the transaction features"}), 400

    missing = [f for f in features if f not in payload or payload[f] is None]
    if len(missing) == len(features):
        return jsonify({"error": "None of the expected features were found", "expected": features}), 422

    proba = float(model.predict_proba(preprocessor.transform(to_frame(payload)))[:, 1][0])
    response = {
        "fraud_probability": proba,
        "is_fraud": proba >= threshold,
        "threshold": threshold,
    }
    if missing:
        response["missing_features"] = missing
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 9696)))
