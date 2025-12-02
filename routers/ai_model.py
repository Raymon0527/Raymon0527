import os
import joblib
import numpy as np
import subprocess
from fastapi import APIRouter
from pydantic import BaseModel

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.dirname(BASE_DIR)               
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler.joblib")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "modeljoblib.py")

# Ensure model exists
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("[INFO] Model/scaler missing — training model using modeljoblib.py...")
    try:
        subprocess.run(
            ["python", TRAIN_SCRIPT],
            check=True,
            cwd=PROJECT_ROOT,
        )
        print("[INFO] Model training completed successfully.")
    except Exception as e:
        print(f"[ERROR] Training script failed: {e}")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Model and scaler loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load model/scaler: {e}")
    model = None
    scaler = None

# Initialize router
router = APIRouter(prefix="/ai", tags=["AI Model"])

class InputData(BaseModel):
    heart_rate: float
    motion_intensity: float

@router.get("/")
def ai_status():
    return {"message": "AI Heart Rate Model API is active"}

@router.post("/predict")
def predict(data: InputData):
    """
    Pure Isolation Forest prediction with 0–100 confidence values.
    Normal + Anomaly always sums to 100, integers only.
    """
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please retrain."}

    try:
        # Convert input
        X = np.array([[data.heart_rate, data.motion_intensity]])
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]

        # Convert score into probability using sigmoid
        prob_normal = 1 / (1 + np.exp(-5 * score))

        confidence_normal = int(round(prob_normal * 100))

        confidence_stress = 100 - confidence_normal

        return {
            "heart_rate": data.heart_rate,
            "motion_intensity": data.motion_intensity,

            "prediction": "STRESSED" if pred == -1 else "NORMAL",
            "anomaly_score": round(float(score), 4),

            "confidence_normal_%": confidence_normal,
            "confidence_stress_%": confidence_stress
        }

    except Exception as e:
        return {"error": str(e)}
