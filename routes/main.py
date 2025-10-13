from fastapi import FastAPI
from app.services.ai_module import detect_anomaly

app = FastAPI(title="Anomaly Detection Demo")

@app.post("/check_heart_rate")
def check_heart_rate(data: dict):
    heart_rate = data["heart_rate"]
    motion = data["motion"]

    # Just print result in console
    detect_anomaly(heart_rate, motion)

    return {"status": "Processed — check console for output."}
