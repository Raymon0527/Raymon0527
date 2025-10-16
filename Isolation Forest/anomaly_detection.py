import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from joblib import dump, load
import os
import time

# --- Configurable settings ---
CSV_FILE = "heart_rate.csv"
MODEL_FILE = "trained_anomaly_model.joblib"
REFRESH_INTERVAL = 5   # seconds between checks
CONTAMINATION = 0.01   # estimated anomaly proportion

# --- Helper function to train a new model ---
def train_model(data):
    X = data[["heart_rate", "motion"]].values
    clf = IsolationForest(contamination=CONTAMINATION, random_state=42)
    clf.fit(X)
    dump(clf, MODEL_FILE)
    print("✅ New model trained and saved.")
    return clf

# --- Load or train model ---
if os.path.exists(MODEL_FILE):
    clf = load(MODEL_FILE)
    print("📦 Loaded existing model.")
else:
    print("⚙️ Training new model (no saved model found)...")
    data_init = pd.read_csv(CSV_FILE)
    clf = train_model(data_init)

# --- Real-time monitoring loop ---
print(f"\n🔁 Monitoring '{CSV_FILE}' for updates every {REFRESH_INTERVAL} seconds...\n")

last_modified = None

while True:
    try:
        # Check if file exists
        if not os.path.exists(CSV_FILE):
            print("❌ CSV file not found. Waiting...")
            time.sleep(REFRESH_INTERVAL)
            continue

        # Detect file changes
        current_modified = os.path.getmtime(CSV_FILE)
        if current_modified != last_modified:
            last_modified = current_modified

            # Load and preprocess latest data
            data = pd.read_csv(CSV_FILE)
            if not {"heart_rate", "motion", "time"}.issubset(data.columns):
                raise ValueError("CSV must contain: time, heart_rate, motion")

            data = data.dropna(subset=["heart_rate", "motion"])
            X = data[["heart_rate", "motion"]].values

            # Predict anomalies
            y_pred = clf.predict(X)
            data["Anomaly"] = np.where(y_pred == -1, "Yes", "No")
            data["Heart Rate Status"] = np.where(data["heart_rate"] > 75, "High", "Normal")

            # Save results to new file
            data.to_csv("heart_rate_with_anomalies.csv", index=False)

            # Print summary
            total_anomalies = (data["Anomaly"] == "Yes").sum()
            print(f"\n🩺 Updated data detected! Total anomalies: {total_anomalies}")

            # --- Visualization ---
            plt.figure(figsize=(12, 6))
            plt.plot(data["time"], data["heart_rate"], label="Heart Rate (BPM)", color="blue")
            plt.plot(data["time"], data["motion"] * 100, label="Motion (scaled x100)", color="green", alpha=0.7)
            plt.scatter(data.loc[data["Anomaly"] == "Yes", "time"],
                        data.loc[data["Anomaly"] == "Yes", "heart_rate"],
                        color="red", label="Anomaly (HR)")
            plt.scatter(data.loc[data["Anomaly"] == "Yes", "time"],
                        data.loc[data["Anomaly"] == "Yes", "motion"] * 100,
                        color="orange", label="Anomaly (Motion)")
            plt.title("Real-Time Heart Rate and Motion Anomaly Detection")
            plt.xlabel("Time")
            plt.ylabel("Signal Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        # Wait before checking again
        time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(REFRESH_INTERVAL)
