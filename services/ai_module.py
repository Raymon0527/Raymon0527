# app/services/ai_module.py
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os

MODEL_PATH = "model/anomaly_detector.pkl"

def train_model():
    """Train the Isolation Forest model and save it."""
    # Example training data (normal heart_rate, motion)
    X_train = np.array([
        [70, 0.1], [72, 0.2], [68, 0.15],
        [75, 0.05], [80, 0.1], [65, 0.25],
        [78, 0.12], [60, 0.18]
    ])
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(" Model trained and saved.")
    return model

# Load model or train if not available
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Loaded existing Isolation Forest model.")
else:
    print("No model found — training a new one.")
    model = train_model()

def detect_anomaly(heart_rate: float, motion: float):
    """Detect anomaly and print the result."""
    features = np.array([[heart_rate, motion]])
    prediction = model.predict(features)
    is_anomaly = prediction[0] == -1

    if is_anomaly:
        print(f" Anomaly Detected! Heart Rate: {heart_rate}, Motion: {motion}")
    else:
        print(f" Normal Reading — Heart Rate: {heart_rate}, Motion: {motion}")

    return is_anomaly
