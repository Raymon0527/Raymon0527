import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# Train model
def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("training_data.csv is missing.")

    df = pd.read_csv(DATA_PATH)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    X = df[["heart_rate", "motion_intensity"]]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IsoForest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return "Model trained successfully."


# Ml Predictions
def predict(heart_rate: float, motion_intensity: float):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data = np.array([[heart_rate, motion_intensity]])
    scaled = scaler.transform(data)

    pred = model.predict(scaled)[0]       
    score = model.decision_function(scaled)[0]

    status = "NORMAL" if pred == 1 else "ANOMALY"

    # Convert anomaly score to confidence %
    confidence_normal = max(0, min(1, (score + 0.5))) * 100
    confidence_anomaly = 100 - confidence_normal

    return {
        "heart_rate": heart_rate,
        "motion_intensity": motion_intensity,

        "isolation_forest_prediction": status,
        "anomaly_score": round(float(score), 4),
        "confidence_normal_%": round(confidence_normal, 2),
        "confidence_anomaly_%": round(confidence_anomaly, 2)
    }


if __name__ == "__main__":
    print(train_model())
