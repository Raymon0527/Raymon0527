import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# File Configuration
CSV_FILE = "cleaned_integrated_heart_motion.csv"   # Integrated CSV file
OUTPUT_FILE = "detected_anomalies.csv"            # Output file name

# Load Integrated CSV File
try:
    data = pd.read_csv(CSV_FILE)
    print(" Integrated CSV file loaded successfully!")
except FileNotFoundError:
    raise SystemExit(
        f" File not found: {CSV_FILE}\n"
        "➡ Please make sure the CSV file is in the same folder as this script."
    )

# Validate Columns
expected_cols = {"heart_rate", "X", "Y", "Z"}  # Only these columns are required
if not expected_cols.issubset(data.columns):
    raise ValueError(f" The file must contain these columns: {expected_cols}")

# Clean Numeric Data
for col in expected_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data.dropna(subset=expected_cols, inplace=True)
print(f" Data cleaned successfully. Total records: {len(data)}")

# Rule-based Heart Rate Anomalies
data["Heart_Anomaly"] = np.where(
    (data["heart_rate"] < 60) | (data["heart_rate"] > 100), "Yes", "No"
)

# AI-based Motion Anomalies (Isolation Forest on X, Y, Z)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data[["X", "Y", "Z"]])
motion_pred = model.predict(data[["X", "Y", "Z"]])
data["Motion_Anomaly"] = np.where(motion_pred == -1, "Yes", "No")

# Combine Heart Rate and Motion Anomalies
data["Anomaly"] = np.where(
    (data["Heart_Anomaly"] == "Yes") | (data["Motion_Anomaly"] == "Yes"),
    "Yes", "No"
)

# Visualization
# Compute motion magnitude for plotting
data["motion_magnitude"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)

plt.figure(figsize=(8, 5))
plt.scatter(
    data.loc[data["Anomaly"] == "No", "heart_rate"],
    data.loc[data["Anomaly"] == "No", "motion_magnitude"],
    color="gray", s=30, alpha=0.6, label="Normal Data"
)
plt.scatter(
    data.loc[data["Anomaly"] == "Yes", "heart_rate"],
    data.loc[data["Anomaly"] == "Yes", "motion_magnitude"],
    color="red", s=60, alpha=0.9, label="Anomalies"
)
plt.title("Heart Rate vs Motion Magnitude — Anomaly Detection", fontsize=12)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Summary and Output
total_anomalies = (data["Anomaly"] == "Yes").sum()
print(f"\nTotal Data Points: {len(data)}")
print(f"Anomalies Detected: {total_anomalies}")

data.to_csv(OUTPUT_FILE, index=False)
print(f" Connected and analyzed data saved to '{OUTPUT_FILE}'")
