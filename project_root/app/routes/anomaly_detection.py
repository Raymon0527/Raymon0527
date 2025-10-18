import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- File Configuration ---
CSV_FILE = "cleaned_integrated_heart_motion.csv"
OUTPUT_FILE = "detected_anomalies.pkl"

# --- Load CSV ---
try:
    data = pd.read_csv(CSV_FILE)
    print(" CSV loaded successfully!")
except FileNotFoundError:
    raise SystemExit(f" File not found: {CSV_FILE}")

# --- Validate Required Columns ---
expected_cols = {"heart_rate", "X", "Y", "Z"}
if not expected_cols.issubset(data.columns):
    raise ValueError(f"File must contain columns: {expected_cols}")

# --- Clean Numeric Data ---
for col in expected_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data.dropna(subset=expected_cols, inplace=True)
print(f"🧹 Data cleaned. Total records: {len(data)}")

# --- Compute Motion Magnitude ---
data["motion_magnitude"] = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2)

# --- Feature Selection for ML ---
features = data[["heart_rate", "X", "Y", "Z"]]

# --- Feature Scaling ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- Train Isolation Forest Model ---
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # 5% anomalies (tune as needed)
    random_state=42
)
data["Anomaly_Label"] = model.fit_predict(scaled_features)

# --- Interpret Model Output ---
data["ML_Anomaly"] = np.where(data["Anomaly_Label"] == -1, "Yes", "No")

# --- Add Rule-Based Detection for Comparison ---
GRAVITY = 1.0
delta_threshold = 0.5
data["net_accel"] = (data["motion_magnitude"] - GRAVITY).abs()
data["Motion_Anomaly"] = np.where(data["net_accel"] >= delta_threshold, "Yes", "No")
data["Heart_Anomaly"] = np.where(
    (data["heart_rate"] <= 60) | (data["heart_rate"] >= 100),
    "Yes", "No"
)

# --- Combine ML + Rule-Based Classification ---
def combined_type(row):
    if row["ML_Anomaly"] == "Yes":
        if row["Heart_Anomaly"] == "Yes" and row["Motion_Anomaly"] == "Yes":
            return "Both (ML+Rule)"
        elif row["Heart_Anomaly"] == "Yes":
            return "Heart (ML+Rule)"
        elif row["Motion_Anomaly"] == "Yes":
            return "Motion (ML+Rule)"
        else:
            return "Anomaly (ML)"
    else:
        return "Normal"

data["Anomaly_Type"] = data.apply(combined_type, axis=1)

# --- Visualization ---
plt.figure(figsize=(12, 7))

plt.scatter(
    data.loc[data["Anomaly_Type"] == "Normal", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Normal", "motion_magnitude"],
    color="lightgreen", s=50, alpha=0.9, label="Normal"
)
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Heart (ML+Rule)", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Heart (ML+Rule)", "motion_magnitude"],
    color="blue", s=60, alpha=0.8, label="Heart Anomaly"
)
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Motion (ML+Rule)", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Motion (ML+Rule)", "motion_magnitude"],
    color="orange", s=60, alpha=0.8, label="Motion Anomaly"
)
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Both (ML+Rule)", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Both (ML+Rule)", "motion_magnitude"],
    color="red", s=80, alpha=0.9, label="Both"
)
plt.scatter(
    data.loc[data["Anomaly_Type"] == "Anomaly (ML)", "heart_rate"],
    data.loc[data["Anomaly_Type"] == "Anomaly (ML)", "motion_magnitude"],
    color="purple", s=70, alpha=0.85, label="ML-Only Anomaly"
)

plt.title("Isolation Forest + Rule-Based Heart & Motion Anomaly Detection", fontsize=14)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Motion Magnitude (g)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# --- Add Count Summary ---
summary = data["Anomaly_Type"].value_counts().fillna(0).astype(int)
table_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
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

# --- Output Results ---
print("\n Anomaly Summary:")
print(summary)
data.to_pickle(OUTPUT_FILE)
print(f"\n Analysis saved to '{OUTPUT_FILE}'")
