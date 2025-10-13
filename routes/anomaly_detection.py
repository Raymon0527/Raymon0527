from fastapi import APIRouter
from app.services.ai_module import detect_anomaly

router = APIRouter()

@router.post("/check_heart_rate")
def check_heart_rate(data: dict):
    heart_rate = data["heart_rate"]
    motion = data["motion"]

    is_anomaly = detect_anomaly(heart_rate, motion)

    return {
        "heart_rate": heart_rate,
        "motion": motion,
        "anomaly_detected": is_anomaly
    }
