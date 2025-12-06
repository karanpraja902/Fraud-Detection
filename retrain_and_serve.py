#!/usr/bin/env python3
"""
Retrains the model with current sklearn version and serves it.
This resolves version compatibility issues.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

def retrain_model():
    """Retrain model with current sklearn version"""
    print("🔄 Retraining model with current sklearn version...")

    # Load data
    df = pd.read_csv('data/processed/train.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train simple model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print(f"Training on {len(X)} samples...")
    model.fit(X, y)

    # Evaluate on test set
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(".3f")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/current_model.pkl')
    print("✅ Model retrained and saved")

    return model

def load_or_retrain_model():
    """Load existing model or retrain if needed"""
    model_path = 'models/current_model.pkl'

    if os.path.exists(model_path):
        try:
            print("Loading existing compatible model...")
            model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    # Retrain if loading fails
    return retrain_model()

# Global model variable
model = None

@app.route('/health', methods=['GET'])
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
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df = pd.DataFrame([data])
        prediction_proba = model.predict_proba(df)
        prediction = model.predict(df)

        result = {
            "prediction": int(prediction[0]),
            "fraud_probability": float(prediction_proba[0][1]),
            "normal_probability": float(prediction_proba[0][0]),
            "is_fraud": bool(prediction[0] == 1)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def info():
    """Info endpoint"""
    return jsonify({
        "service": "fraud-detection-model",
        "status": "retrained with current sklearn",
        "sklearn_version": "1.3.0",
        "model_loaded": model is not None,
        "endpoints": {
            "GET /health": "Health check",
            "GET /": "Service info",
            "POST /predict": "Make prediction"
        }
    })

if __name__ == '__main__':
    model = load_or_retrain_model()

    port = int(os.environ.get('PORT', 3000))
    print(f"🚀 Starting fraud detection service on port {port}")
    print("📊 Model ready for predictions")
    print("🔗 Endpoints:")
    print(f"   GET  http://localhost:{port}/health")
    print(f"   GET  http://localhost:{port}/")
    print(f"   POST http://localhost:{port}/predict")

    app.run(host='0.0.0.0', port=port, debug=False)
