# app/main.py
from fastapi import FastAPI
from app.routes import anomaly_detection

app = FastAPI(
    title="Heart Rate & Motion Anomaly Detection API",
    description="Detects abnormal patterns in heart rate and motion data using Isolation Forest.",
    version="1.0.0"
)

# Register the anomaly detection router
app.include_router(anomaly_detection.router)
