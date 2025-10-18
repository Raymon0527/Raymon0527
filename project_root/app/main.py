from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from app.routes import anomaly_detection
from app import __init__  # just to test import

app = FastAPI()
app.include_router(anomaly_detection.router)

# Load the model once when the server starts
model = joblib.load("detected_anomalies.pkl")  # relative path

# Define the expected input data schema
class Features(BaseModel):
    features: list[float]

# Create a prediction endpoint
@app.post("/predict")
def predict(data: Features):
    input_array = np.array([data.features])  # make 2D array
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
