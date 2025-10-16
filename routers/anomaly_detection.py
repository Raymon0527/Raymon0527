
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -------------------------------
# File Configuration
# -------------------------------
CSV_FILE = "cleaned_integrated_heart_motion.csv"   # Integrated CSV file
OUTPUT_FILE = "connected_anomalies.csv"            # Output file name

# -------------------------------
# Load Integrated CSV File
# -------------------------------
try:
    data = pd.read_csv(CSV_FILE)
    print(" Integrated CSV file loaded successfully!")
except FileNotFoundError:
    raise SystemExit(
        f" File not found: {CSV_FILE}\n"
        "➡ Please make sure 'cleaned_integrated_heart_motion.csv' is in the same folder as this script."
    )

# -------------------------------
# Validate Columns
# -------------------------------
expected_cols = {"heart_rate", "motion"}
if not expected_cols.issubset(data.columns):
    raise ValueError(f" The file must contain these columns: {expected_cols}")

# Clean numeric data
data["heart_rate"] = pd.to_numeric(data["heart_rate"], errors="coerce")
data["motion"] = pd.to_numeric(data["motion"], errors="coerce")
data.dropna(subset=["heart_rate", "motion"], inplace=True)

print(f" Data cleaned successfully. Total records: {len(data)}")

# -------------------------------
# Hybrid Anomaly Detection
# -------------------------------
# Rule-based Heart Rate Anomalies
data["Heart_Anomaly"] = np.where(
    (data["heart_rate"] < 60) | (data["heart_rate"] > 100), "Yes", "No"
)

# AI-based Motion Anomalies (Isolation Forest)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data[["motion"]])
motion_pred = model.predict(data[["motion"]])
data["Motion_Anomaly"] = np.where(motion_pred == -1, "Yes", "No")

# Combine Both
data["Anomaly"] = np.where(
    (data["Heart_Anomaly"] == "Yes") | (data["Motion_Anomaly"] == "Yes"),
    "Yes", "No"
)

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    data.loc[data["Anomaly"] == "No", "heart_rate"],
    data.loc[data["Anomaly"] == "No", "motion"],
    color="gray", s=30, alpha=0.6, label="Normal Data"
)
plt.scatter(
    data.loc[data["Anomaly"] == "Yes", "heart_rate"],
    data.loc[data["Anomaly"] == "Yes", "motion"],
    color="red", s=60, alpha=0.9, label="Anomalies"
)
plt.title("Integrated Heart Rate and Motion — Anomaly Detection", fontsize=12)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------
# Summary and Output
# -------------------------------
total_anomalies = (data["Anomaly"] == "Yes").sum()
print(f"\n Total Data Points: {len(data)}")
print(f" Anomalies Detected: {total_anomalies}")

data.to_csv(OUTPUT_FILE, index=False)
print(f" Connected and analyzed data saved to '{OUTPUT_FILE}'")
