import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
RESULTS_DIR = os.path.join(BASE_DIR, "Analysis", "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "reports", "vision_2030_risk_report.csv")
SPATIAL_PATH = os.path.join(RESULTS_DIR, "spatial", "vision_2030_spatial_intelligence.csv")

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SPATIAL_PATH), exist_ok=True)

# -------- MODELS --------
class RiskANN(nn.Module):
    def __init__(self, input_size):
        super(RiskANN, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x): return self.layers(x)

class SpatialGCN(nn.Module):
    def __init__(self, in_feat):
        super(SpatialGCN, self).__init__()
        self.proj = nn.Linear(in_feat, 64)
        self.spat = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
    def forward(self, x, adj):
        x = F.relu(self.proj(x))
        x = F.relu(self.spat(torch.mm(adj, x)))
        return torch.sigmoid(self.out(x))

ADJACENCY_MAP = {
    'Delhi (UT)': ['Haryana', 'Uttar Pradesh'],
    'Maharashtra': ['Gujarat', 'Madhya Pradesh', 'Chhattisgarh', 'Telangana', 'Karnataka', 'Goa'],
    'Uttar Pradesh': ['Uttarakhand', 'Haryana', 'Delhi (UT)', 'Rajasthan', 'Madhya Pradesh', 'Chhattisgarh', 'Jharkhand', 'Bihar'],
    # Add more key states for regional spillover
}

def generate_production_artifacts():
    print("Synchronizing Production Neural Artifacts (2030 Vision)...")
    
    DATA_PATH = os.path.join(BASE_DIR, r"Analysis\results\tables\DL\lstm_temporal_data.csv")
    df = pd.read_csv(DATA_PATH)
    states = sorted(df['State'].unique())
    features = [c for c in df.columns if c not in ['State', 'Year']]
    
    # 1. GENERATE RISK REPORT (ANN)
    print("   - Calibrating 2030 Neural Risk Audit...")
    df_latest = df[df['Year'] == 2023].sort_values('State')
    sc_std = StandardScaler()
    X_train = sc_std.fit_transform(df[df['Year'] <= 2021][features].values)
    
    # Labeling based on intensity distributions
    df['Intensity'] = df[features].sum(axis=1)
    q75 = df['Intensity'].quantile(0.75)
    y_train = torch.tensor((df[df['Year'] <= 2021]['Intensity'] > q75).astype(float).values, dtype=torch.float32).view(-1, 1)
    
    ann = RiskANN(len(features))
    opt = optim.Adam(ann.parameters(), lr=0.001)
    for _ in range(200):
        out = ann(torch.tensor(X_train, dtype=torch.float32))
        loss = nn.BCELoss()(out, y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    
    X_final = torch.tensor(sc_std.transform(df_latest[features].values), dtype=torch.float32)
    with torch.no_grad():
        risk_scores = ann(X_final).numpy().flatten()
    
    risk_results = []
    for i, state in enumerate(states):
        score = float(risk_scores[i])
        grade = "CRITICAL" if score > 0.8 else ("HIGH" if score > 0.5 else "STABLE")
        risk_results.append({'State': state, 'Risk_Grade': grade, 'Risk_Score': score})
    
    pd.DataFrame(risk_results).to_csv(REPORT_PATH, index=False)

    # 2. GENERATE SPATIAL INDEX (GNN)
    print("   - Mapping 2030 Regional Spillover Index...")
    state_to_idx = {s: i for i, s in enumerate(states)}
    adj = torch.eye(len(states))
    for s, neighbors in ADJACENCY_MAP.items():
        if s in state_to_idx:
            u = state_to_idx[s]
            for n in neighbors:
                if n in state_to_idx: adj[u, state_to_idx[n]] = 1.0
    
    gnn = SpatialGCN(len(features))
    sc_minmax = MinMaxScaler()
    feat_t = torch.tensor(sc_minmax.fit_transform(df_latest[features].values), dtype=torch.float32)
    
    with torch.no_grad():
        spat_scores = gnn(feat_t, adj).numpy().flatten()
    
    spatial_results = []
    for i, state in enumerate(states):
        spatial_results.append({'State': state, 'Spatial_Spillover_Index': float(spat_scores[i])})
    
    pd.DataFrame(spatial_results).to_csv(SPATIAL_PATH, index=False)

    print("\nPRODUCTION SYNC COMPLETE")
    print(f"Risk Report: {REPORT_PATH}")
    print(f"Spatial Logic: {SPATIAL_PATH}")

if __name__ == "__main__":
    generate_production_artifacts()
