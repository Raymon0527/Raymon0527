# app/routes/anomaly_detection.py
from fastapi import APIRouter
from app.services.ai_module import detect_anomaly, add_training_data, retrain_model

router = APIRouter(
    prefix="/anomaly",      # group all endpoints under /anomaly
    tags=["Anomaly Detection"]
)

@router.post("/check")
def check_anomaly(data: dict):
    """
    Check if the given heart rate and motion values indicate an anomaly.
    Automatically logs data for retraining.
    """
    heart_rate = data["heart_rate"]
    motion = data["motion"]

    is_anomaly = detect_anomaly(heart_rate, motion)

    # Store data for future retraining
    add_training_data(heart_rate, motion)

    return {
        "heart_rate": heart_rate,
        "motion": motion,
        "anomaly_detected": is_anomaly
    }

@router.post("/retrain")
def retrain_anomaly_model():
    """
    Retrain the Isolation Forest model using stored training data.
    """
    retrain_model()
    return {"message": " Model retrained successfully using latest data."}
