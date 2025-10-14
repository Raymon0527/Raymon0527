import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler

import xlwings as xw

from datetime import datetime

class ExcelAnomalyDetector:

    def __init__(self, workbook_path, sheet_name):

        self.workbook_path = workbook_path

        self.sheet_name = sheet_name

        self.wb = xw.Book(workbook_path)

        self.ws = self.wb.sheets[sheet_name]

        self.scaler = StandardScaler()

        

    def load_data_from_excel(self, data_range):

        """Load data from Excel range for analysis"""

        data = self.ws.range(data_range).value

        

        # Convert to DataFrame

        headers = data[0]

        values = data[1:]

        

        df = pd.DataFrame(values, columns=headers)

        

        # Handle missing values and data types

        df = df.dropna()

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        

        return df, numeric_columns

    

    def detect_financial_anomalies(self, data_range, contamination=0.1):

        """Detect anomalies in financial data using Isolation Forest"""

        df, numeric_cols = self.load_data_from_excel(data_range)

        

        # Prepare features for anomaly detection

        features = df[numeric_cols].values

        features_scaled = self.scaler.fit_transform(features)

        

        # Train Isolation Forest model

        iso_forest = IsolationForest(

            contamination=contamination,

            random_state=42,

            n_estimators=100

        )

        

        # Predict anomalies (-1 = anomaly, 1 = normal)

        predictions = iso_forest.fit_predict(features_scaled)

        anomaly_scores = iso_forest.decision_function(features_scaled)

        

        # Add results to DataFrame

        df['anomaly_flag'] = predictions

        df['anomaly_score'] = anomaly_scores

        df['risk_level'] = self.categorize_risk_levels(anomaly_scores)

        

        return df

    

    def categorize_risk_levels(self, scores):

        """Categorize anomaly scores into risk levels"""

        risk_levels = []

        for score in scores:

            if score < -0.5:

                risk_levels.append('HIGH')

            elif score < -0.2:

                risk_levels.append('MEDIUM')

            elif score < 0:

                risk_levels.append('LOW')

            else:

                risk_levels.append('NORMAL')

        return risk_levels

    

    def highlight_anomalies_in_excel(self, results_df, start_row=2):

        """Apply visual highlighting to anomalies in Excel"""

        

        # Clear existing formatting

        data_range = f"A{start_row}:Z{start_row + len(results_df)}"

        self.ws.range(data_range).color = None

        

        # Apply conditional formatting based on risk levels

        for i, (_, row) in enumerate(results_df.iterrows()):

            excel_row = start_row + i

            

            if row['anomaly_flag'] == -1:  # Anomaly detected

                if row['risk_level'] == 'HIGH':

                    # Dark red for high risk

                    self.ws.range(f"A{excel_row}:H{excel_row}").color = (255, 200, 200)

                elif row['risk_level'] == 'MEDIUM':

                    # Light red for medium risk

                    self.ws.range(f"A{excel_row}:H{excel_row}").color = (255, 230, 230)

                elif row['risk_level'] == 'LOW':

                    # Light yellow for low risk

                    self.ws.range(f"A{excel_row}:H{excel_row}").color = (255, 255, 200)

        

        # Add anomaly indicators in dedicated columns

        anomaly_col = 'I'

        score_col = 'J'

        risk_col = 'K'

        

        # Headers

        self.ws.range(f"{anomaly_col}1").value = "Anomaly"

        self.ws.range(f"{score_col}1").value = "Score"

        self.ws.range(f"{risk_col}1").value = "Risk Level"

        

        # Data

        for i, (_, row) in enumerate(results_df.iterrows()):

            excel_row = start_row + i

            self.ws.range(f"{anomaly_col}{excel_row}").value = "⚠️" if row['anomaly_flag'] == -1 else "✓"

            self.ws.range(f"{score_col}{excel_row}").value = round(row['anomaly_score'], 3)

            self.ws.range(f"{risk_col}{excel_row}").value = row['risk_level']

    

    def generate_anomaly_summary(self, results_df):

        """Generate summary statistics for anomaly detection"""

        total_records = len(results_df)

        anomalies_detected = len(results_df[results_df['anomaly_flag'] == -1])

        

        risk_summary = results_df['risk_level'].value_counts()

        

        summary = {

            'total_records': total_records,

            'anomalies_detected': anomalies_detected,

            'anomaly_rate': (anomalies_detected / total_records) * 100,

            'risk_distribution': risk_summary.to_dict()

        }

        

        # Write summary to Excel

        self.ws.range('M1').value = "Anomaly Detection Summary"

        self.ws.range('M2').value = f"Total Records: {total_records}"

        self.ws.range('M3').value = f"Anomalies Detected: {anomalies_detected}"

        self.ws.range('M4').value = f"Anomaly Rate: {summary['anomaly_rate']:.2f}%"

        

        return summary

# Usage example for expense analysis

def analyze_expense_anomalies():

    """Analyze expense data for fraudulent transactions"""

    

    detector = ExcelAnomalyDetector('expense_data.xlsx', 'Transactions')

    

    # Detect anomalies in expense data

    results = detector.detect_financial_anomalies('A1:H1000', contamination=0.05)

    

    # Highlight anomalies in Excel

    detector.highlight_anomalies_in_excel(results)

    

    # Generate summary report

    summary = detector.generate_anomaly_summary(results)

    

    print(f"Analysis complete: {summary['anomalies_detected']} anomalies detected")

    print(f"Risk distribution: {summary['risk_distribution']}")

    

    return results

# Advanced multi-feature anomaly detection

def detect_complex_patterns():

    """Detect complex multi-dimensional anomalies"""

    

    detector = ExcelAnomalyDetector('operations_data.xlsx', 'KPI_Dashboard')

    

    # Load operational data

    df, numeric_cols = detector.load_data_from_excel('A1:J500')

    

    # Create additional features for pattern detection

    df['revenue_per_employee'] = df['revenue'] / df['employee_count']

    df['efficiency_ratio'] = df['output'] / df['input_cost']

    df['growth_rate'] = df['current_month'] / df['previous_month'] - 1

    

    # Update numeric columns to include new features

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    

    # Multi-model ensemble approach

    features = df[numeric_cols].values

    features_scaled = detector.scaler.fit_transform(features)

    

    # Primary Isolation Forest

    iso_forest = IsolationForest(contamination=0.08, random_state=42)

    iso_predictions = iso_forest.fit_predict(features_scaled)

    

    # Secondary model for validation

    from sklearn.ensemble import LocalOutlierFactor

    lof = LocalOutlierFactor(contamination=0.08)

    lof_predictions = lof.fit_predict(features_scaled)

    

    # Combine predictions (consensus approach)

    combined_predictions = []

    for i in range(len(iso_predictions)):

        if iso_predictions[i] == -1 and lof_predictions[i] == -1:

            combined_predictions.append(-1)  # High confidence anomaly

        elif iso_predictions[i] == -1 or lof_predictions[i] == -1:

            combined_predictions.append(0)   # Medium confidence anomaly

        else:

            combined_predictions.append(1)   # Normal

    

    df['consensus_anomaly'] = combined_predictions

    

    return df

