import os
import joblib
import numpy as np
import subprocess
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ai", tags=["AI Model"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "modeljoblib.py")

# Ensure the model exists
if not os.path.exists(MODEL_PATH):
    print(" Model not found — training with modeljoblib.py...")
    try:
        subprocess.run(["python", TRAIN_SCRIPT], check=True, cwd=BASE_DIR)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Training failed: {e}")
else:
    print(" Model found — loading...")

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f" Model loaded from: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define Input Data
class InputData(BaseModel):
    heart_rate: float
    motion_intensity: float

@router.get("/")
def ai_status():
    """Check if AI model is running"""
    return {"message": "AI Model for Heart Rate Anomaly Detection is active"}

@router.post("/predict")
def predict(data: InputData):
    """
    Predict if the given data indicates a NORMAL or STRESSED state.
    """
    try:
        motion_map = {"low": 0, "moderate": 1, "high": 2}
        motion_encoded = motion_map.get(data.motion.lower(), 0)

        features = np.array([[data.heart_rate, motion_encoded, data.ir_value]])
        prediction = model.predict(features)[0]

        result = "STRESSED" if prediction == -1 else "NORMAL"

        return {
            "heart_rate": data.heart_rate,
            "motion_intensity": data.motion_intensity,
            "stress_intensity": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
