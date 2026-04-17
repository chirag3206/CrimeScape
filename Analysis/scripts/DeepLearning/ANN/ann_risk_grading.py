

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Paths
HISTORICAL_DATA = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\tables\DL\lstm_temporal_data.csv"
FORECAST_2030 = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\forecasts\forecast_2030.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\reports"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- ANN Architecture ---
class RiskANN(nn.Module):
    def __init__(self, input_size):
        super(RiskANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Risk Score (0 to 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

def run_ann_grading():
    print("Initializing Deep Risk Auditor (ANN)...")
    
    # 1. Load Data
    historical_df = pd.read_csv(HISTORICAL_DATA)
    forecast_df = pd.read_csv(FORECAST_2030)
    
    features = [c for c in historical_df.columns if c not in ['State', 'Year']]
    
    # 2. Define "Risk Labels" for training (Unsupervised to Supervised Proxy)
    # We define Risk based on whether a state's total crime rate is above the 75th percentile
    crime_sums = historical_df[features].sum(axis=1)
    threshold = crime_sums.quantile(0.75)
    historical_df['Risk_Label'] = (crime_sums > threshold).astype(float)
    
    # 3. Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(historical_df[features])
    y = historical_df['Risk_Label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # 4. Training
    model = RiskANN(len(features))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print("   - Training Risk-Pattern Detection Layers...")
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # 5. Inference & Trend Velocity Calibration
    print("   - Auditing the 2030 Horizon with Momentum Analysis...")
    X_2030_full = torch.tensor(scaler.transform(forecast_df[features]), dtype=torch.float32)
    with torch.no_grad():
        all_risk_scores = model(X_2030_full).numpy().flatten()
    
    forecast_df['Base_Risk_Score'] = all_risk_scores
    
    # Calculate Velocity using Synchronized Dashboard Logic
    # We use the same 'Per 100k' weighted sum as app.py
    rate_cols = ["Children_Total_All_R", "Women_Total_Crime_R"]
    count_cols = ["Juvenile_Total_Cognizable", "Human Traficking_GrandTotal"]
    
    # Pre-calculate normalized intensity per row
    forecast_df['Standard_Intensity'] = (
        forecast_df[[c for c in rate_cols if c in forecast_df.columns]].fillna(0).sum(axis=1) +
        (forecast_df[[c for c in count_cols if c in forecast_df.columns]].fillna(0).sum(axis=1) / 55.0)
    )
    
    velocity_map = {}
    for state in forecast_df['State'].unique():
        state_data = forecast_df[forecast_df['State'] == state]
        try:
            val_2024 = state_data[state_data['Year'] == 2024]['Standard_Intensity'].values[0]
            val_2030 = state_data[state_data['Year'] == 2030]['Standard_Intensity'].values[0]
            velocity = (val_2030 - val_2024) / (val_2024 + 1e-6)
            velocity_map[state] = velocity
        except:
            velocity_map[state] = 0

    forecast_df['Trend_Velocity'] = forecast_df['State'].map(velocity_map)
    
    # 6. Generate Hybrid Risk Index
    # Hybrid = (70% Neural Magnitude) + (40% Trend Momentum)
    # Velocity is boosted to ensure it impacts the grade significantly
    forecast_df['Hybrid_Risk_Score'] = (
        (forecast_df['Base_Risk_Score'] * 0.6) + 
        (forecast_df['Trend_Velocity'].clip(-0.5, 0.5) * 0.8) # 50% growth adds 0.4 to score
    ).clip(0, 1)
    
    # Define Dynamic Tiers: More aggressive toward positive momentum
    def grade_risk(score, velocity):
        # Escalate any state with > 8% growth automatically
        if score > 0.7 or velocity > 0.15: return "CRITICAL"
        if score > 0.4 or velocity > 0.08: return "HIGH"
        if score > 0.15 or velocity > 0.03: return "STABLE"
        return "LOW"
    
    forecast_df['Risk_Grade'] = forecast_df.apply(lambda x: grade_risk(x['Hybrid_Risk_Score'], x['Trend_Velocity']), axis=1)
    
    # 7. Save Synchronized Report
    output_file = os.path.join(OUTPUT_DIR, "vision_2030_risk_report.csv")
    final_report = forecast_df[forecast_df['Year'] == 2030][['State', 'Year', 'Hybrid_Risk_Score', 'Trend_Velocity', 'Risk_Grade']]
    final_report = final_report.rename(columns={'Hybrid_Risk_Score': 'Risk_Score'})
    final_report = final_report.sort_values('Risk_Score', ascending=False)
    
    final_report.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("VISION 2030 RISK AUDIT COMPLETE")
    print(f"Report saved: {output_file}")
    print("Summary of 2030 Critical States:")
    print(final_report[final_report['Risk_Grade'] == 'CRITICAL']['State'].tolist())
    print("="*40)

if __name__ == "__main__":
    run_ann_grading()
