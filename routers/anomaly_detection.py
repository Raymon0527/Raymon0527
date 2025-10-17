import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# File Configuration
CSV_FILE = "cleaned_integrated_heart_motion.csv"
OUTPUT_FILE = "detected_anomalies.csv"

# Load CSV
try:
    data = pd.read_csv(CSV_FILE)
    print(" Integrated CSV loaded successfully!")
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

# Heart Rate Anomalies (Dynamic)
hr_mean = data["heart_rate"].mean()
hr_std = data["heart_rate"].std()
lower_bound = hr_mean - 2 * hr_std
upper_bound = hr_mean + 2 * hr_std

data["Heart_Anomaly"] = np.where(
    (data["heart_rate"] < lower_bound) | (data["heart_rate"] > upper_bound), "Yes", "No"
)
print(f" Heart rate anomalies detected using mean±2σ ({lower_bound:.1f}-{upper_bound:.1f} BPM)")

# Motion Anomalies (Isolation Forest)
motion_features = data[["X", "Y", "Z"]].copy()

# Scale data for better detection
scaler = StandardScaler()
motion_scaled = scaler.fit_transform(motion_features)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(motion_scaled)
motion_pred = model.predict(motion_scaled)
data["Motion_Anomaly"] = np.where(motion_pred == -1, "Yes", "No")
print(" Motion anomalies detected with Isolation Forest")

# Combined Anomalies
data["Anomaly"] = np.where(
    (data["Heart_Anomaly"] == "Yes") | (data["Motion_Anomaly"] == "Yes"), "Yes", "No"
)

# Visualization
data["motion_magnitude"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)

plt.figure(figsize=(10, 6))

# Normal Data
plt.scatter(
    data.loc[data["Anomaly"] == "No", "heart_rate"],
    data.loc[data["Anomaly"] == "No", "motion_magnitude"],
    color="gray", s=30, alpha=0.6, label="Normal"
)

# Heart Rate Only Anomalies
plt.scatter(
    data.loc[(data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "No"), "heart_rate"],
    data.loc[(data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "No"), "motion_magnitude"],
    color="blue", s=50, alpha=0.8, label="Heart Rate Anomaly"
)

# Motion Only Anomalies
plt.scatter(
    data.loc[(data["Heart_Anomaly"] == "No") & (data["Motion_Anomaly"] == "Yes"), "heart_rate"],
    data.loc[(data["Heart_Anomaly"] == "No") & (data["Motion_Anomaly"] == "Yes"), "motion_magnitude"],
    color="orange", s=50, alpha=0.8, label="Motion Anomaly"
)

# Both Anomalies
plt.scatter(
    data.loc[(data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "Yes"), "heart_rate"],
    data.loc[(data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "Yes"), "motion_magnitude"],
    color="red", s=70, alpha=0.9, label="Combined Anomaly"
)

plt.title("Heart Rate vs Motion — Anomaly Detection", fontsize=14)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Summary & Output
total_anomalies = (data["Anomaly"] == "Yes").sum()
hr_only = ((data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "No")).sum()
motion_only = ((data["Heart_Anomaly"] == "No") & (data["Motion_Anomaly"] == "Yes")).sum()
both = ((data["Heart_Anomaly"] == "Yes") & (data["Motion_Anomaly"] == "Yes")).sum()

print(f"\nTotal Records: {len(data)}")
print(f"Total Anomalies Detected: {total_anomalies}")
print(f" - Heart Rate Only: {hr_only}")
print(f" - Motion Only: {motion_only}")
print(f" - Both: {both}")

# Save CSV
data.to_csv(OUTPUT_FILE, index=False)
print(f" Analysis saved to '{OUTPUT_FILE}'")