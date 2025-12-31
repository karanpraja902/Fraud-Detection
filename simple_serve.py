#!/usr/bin/env python3
"""
Simple model serving script for fraud detection.
This bypasses MLflow version compatibility issues.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load the model directly from joblib file"""
    global model

    # Try different possible model locations
    possible_paths = [
        "models/fraud-detector/model.pkl",
        "models/latest_model.pkl",
        "models/model.pkl"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model from: {path}")
                model = joblib.load(path)
                print(f"✅ Model loaded successfully: {type(model)}")
                return True
            except Exception as e:
                print(f"❌ Failed to load model from {path}: {e}")
                continue

    print("❌ Could not load model from any location")
    return False

@app.route('/health', methods=['GET'])
@app.route('/healthz', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction_proba = model.predict_proba(df)
        prediction = model.predict(df)

        # Return result
        result = {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0][1]),  # Probability of fraud (class 1)
            "fraud_probability": float(prediction_proba[0][1]),
            "normal_probability": float(prediction_proba[0][0])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def info():
    """Info endpoint"""
    return jsonify({
        "service": "fraud-detection-model",
        "version": "1.0",
        "model_loaded": model is not None,
        "endpoints": {
            "GET /health": "Health check",
            "GET /": "Service info",
            "POST /predict": "Make prediction"
        },
        "prediction_format": {
            "V1": "float",
            "V2": "float",
            "V3": "float",
            "V4": "float",
            "V5": "float",
            "Amount": "float",
            "...": "other features"
        }
    })

if __name__ == '__main__':
    if load_model():
        port = int(os.environ.get('PORT', 3000))
        print(f"🚀 Starting fraud detection service on port {port}")
        print("📊 Model loaded and ready for predictions")
        print("🔗 Endpoints:")
        print(f"   GET  http://localhost:{port}/health")
        print(f"   GET  http://localhost:{port}/")
        print(f"   POST http://localhost:{port}/predict")
        print("\nExample prediction request:")
        print('curl -X POST http://localhost:3000/predict \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"V1": -1.36, "V2": -0.0728, "V3": 2.54, "V4": 1.38, "Amount": 149.62}\'')

        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Failed to load model. Exiting.")
        exit(1)
