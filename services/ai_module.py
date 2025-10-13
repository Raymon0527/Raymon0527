# app/services/ai_module.py
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os

MODEL_PATH = "model/anomaly_detector.pkl"
DATA_PATH = "model/training_data.npy"

# ✅ Load or initialize model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️ No saved model found — training new IsolationForest model.")
    X_train = np.array([[70, 0.1], [72, 0.2], [75, 0.05], [80, 0.1]])
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    joblib.dump(model, MODEL_PATH)

# ✅ Load or initialize training data
if os.path.exists(DATA_PATH):
    X_data = np.load(DATA_PATH)
else:
    X_data = np.array([[70, 0.1], [72, 0.2], [75, 0.05], [80, 0.1]])
    np.save(DATA_PATH, X_data)

def detect_anomaly(heart_rate: float, motion: float) -> bool:
    """Return True if anomaly detected, else False."""
    features = np.array([[heart_rate, motion]])
    prediction = model.predict(features)
    return prediction[0] == -1  # -1 = anomaly

def add_training_data(heart_rate: float, motion: float):
    """Append a new sample to the dataset."""
    global X_data
    new_sample = np.array([[heart_rate, motion]])
    X_data = np.vstack([X_data, new_sample])
    np.save(DATA_PATH, X_data)

def retrain_model():
    """Retrain the Isolation Forest model using the updated dataset."""
    global model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_data)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model retrained on {len(X_data)} samples.")
