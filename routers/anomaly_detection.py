import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File Configuration
CSV_FILE = "cleaned_integrated_heart_motion.csv"
OUTPUT_FILE = "detected_anomalies.csv"

# Load CSV
try:
    data = pd.read_csv(CSV_FILE)
    print(" CSV loaded successfully!")
except FileNotFoundError:
    raise SystemExit(f" File not found: {CSV_FILE}")

# Validate Columns
expected_cols = {"heart_rate", "X", "Y", "Z"}
if not expected_cols.issubset(data.columns):
    raise ValueError(f"File must contain columns: {expected_cols}")

# Clean Numeric Data
for col in expected_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data.dropna(subset=expected_cols, inplace=True)
print(f" Data cleaned. Total records: {len(data)}")

# Rule-Based Heart Rate Anomaly
data["Heart_Anomaly"] = np.where(
    (data["heart_rate"] <= 60) | (data["heart_rate"] >= 100),
    "Yes", "No"
)

# Rule-Based Motion Anomaly
# Threshold: 1.5 g
motion_threshold = 1.5
data["Motion_Anomaly"] = np.where(
    (data["X"].abs() >= motion_threshold) |
    (data["Y"].abs() >= motion_threshold) |
    (data["Z"].abs() >= motion_threshold),
    "Yes", "No"
)

# Combined Anomaly
data["Anomaly"] = np.where(
    (data["Heart_Anomaly"] == "Yes") | (data["Motion_Anomaly"] == "Yes"),
    "Yes", "No"
)

# Compute Motion Magnitude
data["motion_magnitude"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)

# Visualization
plt.figure(figsize=(10, 6))

# Normal Data
plt.scatter(
    data.loc[data["Anomaly"] == "No", "heart_rate"],
    data.loc[data["Anomaly"] == "No", "motion_magnitude"],
    color="gray", s=30, alpha=0.6, label="Normal"
)

# Anomalies
plt.scatter(
    data.loc[data["Anomaly"] == "Yes", "heart_rate"],
    data.loc[data["Anomaly"] == "Yes", "motion_magnitude"],
    color="red", s=60, alpha=0.9, label="Anomaly"
)

plt.title("Heart Rate & Motion Rule-Based Anomaly Detection", fontsize=14)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude (g)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Summary & Output# -----------------------------
total_anomalies = (data["Anomaly"] == "Yes").sum()
print(f"\nTotal Records: {len(data)}")
print(f"Total Anomalies Detected: {total_anomalies}")
print(f" - Heart Rate Anomalies: {(data['Heart_Anomaly'] == 'Yes').sum()}")
print(f" - Motion Anomalies: {(data['Motion_Anomaly'] == 'Yes').sum()}")

# Save CSV
data.to_csv(OUTPUT_FILE, index=False)
print(f" Analysis saved to '{OUTPUT_FILE}'")
