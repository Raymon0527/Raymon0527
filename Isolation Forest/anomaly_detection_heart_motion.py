import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd

# --- Generate simulated correlated heart rate and motion data ---
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

# Simulate heart rate (resting 60–75 bpm, active 75–120 bpm)
heart_rate = 60 + 30 * np.abs(np.sin(2 * np.pi * time / 100)) + np.random.normal(0, 2, n_samples)

# Determine "high" heart rate (above 75 bpm)
is_high_hr = heart_rate > 75

# Motion corresponds to heart rate:
# Higher heart rate means more motion; lower means less
motion = np.where(is_high_hr,
                  np.random.uniform(0.6, 1.0, n_samples),   # active movement
                  np.random.uniform(0.0, 0.5, n_samples))   # low movement
motion += np.random.normal(0, 0.05, n_samples)              # slight noise

# --- Introduce anomalies (irregular or mismatched patterns) ---
# Example: high heart rate but low motion, or low heart rate but high motion
anomaly_indices = [200, 500, 750]
for i in anomaly_indices:
    if heart_rate[i] > 75:
        motion[i] = np.random.uniform(0.0, 0.2)  # unrealistic low motion despite high HR
    else:
        motion[i] = np.random.uniform(0.8, 1.0)  # unrealistic high motion despite low HR
    heart_rate[i] += np.random.uniform(20, 40)    # abnormal HR spike

# --- Combine both signals ---
X = np.column_stack((heart_rate, motion))

# --- Apply Isolation Forest for anomaly detection ---
clf = IsolationForest(contamination=0.01, random_state=42)
y_pred = clf.fit_predict(X)

# --- Visualization ---
plt.figure(figsize=(12, 6))

# Plot heart rate and motion together
plt.plot(time, heart_rate, label='Heart Rate (BPM)', color='blue')
plt.plot(time, motion * 100, label='Motion (scaled x100)', color='green', alpha=0.7)

# Highlight detected anomalies
plt.scatter(time[y_pred == -1], heart_rate[y_pred == -1], color='red', label='Anomaly (HR)')
plt.scatter(time[y_pred == -1], motion[y_pred == -1] * 100, color='orange', label='Anomaly (Motion)')

plt.title('Heart Rate and Motion Correlation with Detected Anomalies')
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print total anomalies detected ---
print(f"Number of detected anomalies: {sum(y_pred == -1)}")

# --- Merged data table ---
data = pd.DataFrame({
    "Time": time,
    "Heart Rate (BPM)": heart_rate,
    "Motion Level": motion,
    "Heart Rate Status": ["High" if hr > 75 else "Normal" for hr in heart_rate],
    "Anomaly": ["Yes" if p == -1 else "No" for p in y_pred]
})

print("\nMerged Data Table (first 10 rows):")
print(data.head(10))
