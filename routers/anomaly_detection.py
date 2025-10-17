import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File Configuration
CSV_FILE = "cleaned_integrated_heart_motion.csv"
OUTPUT_FILE = "detected_anomalies.csv"

# Load CSV
try:
    data = pd.read_csv(CSV_FILE)
    print("CSV loaded successfully!")
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

# Corrected Anomaly Classification
def anomaly_type(row):
    if row["Heart_Anomaly"] == "Yes" and row["Motion_Anomaly"] == "Yes":
        return "Both"
    elif row["Heart_Anomaly"] == "Yes" and row["Motion_Anomaly"] == "No":
        return "Heart Only"
    elif row["Heart_Anomaly"] == "No" and row["Motion_Anomaly"] == "Yes":
        return "Motion Only"
    else:
        return "Normal"

data["Anomaly_Type"] = data.apply(anomaly_type, axis=1)

# Compute Motion Magnitude
data["motion_magnitude"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)

# Visualization with Count Table
plt.figure(figsize=(10, 6))

# Normal
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Normal", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Normal", "motion_magnitude"],
    color="lightgreen", s=50, alpha=0.9, label="Normal"
)

# Heart Only
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Heart Only", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Heart Only", "motion_magnitude"],
    color="blue", s=60, alpha=0.8, label="Heart Only"
)

# Motion Only
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Motion Only", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Motion Only", "motion_magnitude"],
    color="orange", s=60, alpha=0.8, label="Motion Only"
)

# Both Anomalies
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Both", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Both", "motion_magnitude"],
    color="red", s=80, alpha=0.9, label="Both"
)

plt.title("Heart Rate & Motion Anomaly Detection", fontsize=14)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude (g)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Add Count Table
summary = data["Anomaly_Type"].value_counts()
table_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])

# Place the table in the top-right corner
plt.gca().text(
    0.98, 0.98, table_text,
    horizontalalignment='right',
    verticalalignment='top',
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

plt.tight_layout()
plt.show()

# Summary & Output
summary = data["Anomaly_Type"].value_counts()
print("\nAnomaly Summary:")
print(summary)

# Save CSV
data.to_csv(OUTPUT_FILE, index=False)
print(f"\n Analysis saved to '{OUTPUT_FILE}'")