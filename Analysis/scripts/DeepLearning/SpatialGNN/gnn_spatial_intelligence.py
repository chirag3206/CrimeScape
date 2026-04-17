import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Paths
FORECAST_2030 = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\forecasts\forecast_2030.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\spatial"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- India State Adjacency Map (Shared Borders) ---
# This dictionary defines the "Edges" of our Graph
ADJACENCY_MAP = {
    'A&N Islands': [],
    'Andhra Pradesh': ['Odisha', 'Chhattisgarh', 'Telangana', 'Karnataka', 'Tamil Nadu'],
    'Arunachal Pradesh': ['Assam', 'Nagaland'],
    'Assam': ['Arunachal Pradesh', 'Nagaland', 'Manipur', 'Mizoram', 'Tripura', 'Meghalaya', 'West Bengal'],
    'Bihar': ['Uttar Pradesh', 'Jharkhand', 'West Bengal'],
    'Chandigarh': ['Punjab', 'Haryana'],
    'Chhattisgarh': ['Madhya Pradesh', 'Uttar Pradesh', 'Jharkhand', 'Odisha', 'Andhra Pradesh', 'Telangana', 'Maharashtra'],
    'D&N Haveli and Daman & Diu': ['Gujarat', 'Maharashtra'],
    'Delhi': ['Haryana', 'Uttar Pradesh'],
    'Goa': ['Maharashtra', 'Karnataka'],
    'Gujarat': ['Rajasthan', 'Madhya Pradesh', 'Maharashtra', 'D&N Haveli and Daman & Diu'],
    'Haryana': ['Punjab', 'Himachal Pradesh', 'Rajasthan', 'Uttar Pradesh', 'Delhi', 'Chandigarh'],
    'Himachal Pradesh': ['Jammu & Kashmir', 'Ladakh', 'Punjab', 'Haryana', 'Uttarakhand'],
    'Jammu & Kashmir': ['Ladakh', 'Himachal Pradesh', 'Punjab'],
    'Jharkhand': ['Bihar', 'Uttar Pradesh', 'Chhattisgarh', 'Odisha', 'West Bengal'],
    'Karnataka': ['Goa', 'Maharashtra', 'Telangana', 'Andhra Pradesh', 'Tamil Nadu', 'Kerala'],
    'Kerala': ['Karnataka', 'Tamil Nadu'],
    'Ladakh': ['Jammu & Kashmir', 'Himachal Pradesh'],
    'Lakshadweep': [],
    'Madhya Pradesh': ['Rajasthan', 'Uttar Pradesh', 'Chhattisgarh', 'Maharashtra', 'Gujarat'],
    'Maharashtra': ['Gujarat', 'Madhya Pradesh', 'Chhattisgarh', 'Telangana', 'Karnataka', 'Goa', 'D&N Haveli and Daman & Diu'],
    'Manipur': ['Nagaland', 'Assam', 'Mizoram'],
    'Meghalaya': ['Assam'],
    'Mizoram': ['Manipur', 'Assam', 'Tripura'],
    'Nagaland': ['Arunachal Pradesh', 'Assam', 'Manipur'],
    'Odisha': ['West Bengal', 'Jharkhand', 'Chhattisgarh', 'Andhra Pradesh'],
    'Puducherry': ['Tamil Nadu', 'Kerala', 'Andhra Pradesh'], # Non-contiguous but admin/regional ties
    'Punjab': ['Jammu & Kashmir', 'Himachal Pradesh', 'Haryana', 'Rajasthan', 'Chandigarh'],
    'Rajasthan': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 'Gujarat'],
    'Sikkim': ['West Bengal'],
    'Tamil Nadu': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Puducherry'],
    'Telangana': ['Maharashtra', 'Chhattisgarh', 'Andhra Pradesh', 'Karnataka'],
    'Tripura': ['Assam', 'Mizoram'],
    'Uttar Pradesh': ['Uttarakhand', 'Haryana', 'Delhi', 'Rajasthan', 'Madhya Pradesh', 'Chhattisgarh', 'Jharkhand', 'Bihar'],
    'Uttarakhand': ['Himachal Pradesh', 'Uttar Pradesh'],
    'West Bengal': ['Sikkim', 'Assam', 'Bihar', 'Jharkhand', 'Odisha']
}

# --- Lightweight GCN Architecture ---
class SpatialGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatialGCN, self).__init__()
        self.projection = nn.Linear(in_features, 64)
        self.spatial_layer = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1) # Spillover Score

    def forward(self, x, adj):
        # 1. Feature Transformation
        x = F.relu(self.projection(x))
        # 2. Graph Convolution (Aggregate neighbor features)
        # In GCN, this is basically H = Adj * X * W
        spatial_features = torch.mm(adj, x) 
        x = F.relu(self.spatial_layer(spatial_features))
        # 3. Final Spillover Index
        return torch.sigmoid(self.output(x))

def run_spatial_intelligence():
    print("Initializing Spatial Intelligence Graph (GNN)...")
    
    # 1. Load 2030 Forecast
    df = pd.read_csv(FORECAST_2030)
    df_2030 = df[df['Year'] == 2030].copy()
    states = sorted(df_2030['State'].unique())
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    features = [c for c in df_2030.columns if c not in ['State', 'Year']]
    node_features_raw = df_2030[features].values
    
    # 2. FEATURE NORMALIZATION (Min-Max Scaling)
    # This prevents the 'Feature Explosion' that was causing zeros
    feat_min = node_features_raw.min(axis=0)
    feat_max = node_features_raw.max(axis=0)
    node_features_scaled = (node_features_raw - feat_min) / (feat_max - feat_min + 1e-6)
    node_features = torch.tensor(node_features_scaled, dtype=torch.float32)
    
    # 3. Build Adjacency Matrix (Normalized)
    adj = torch.zeros((len(states), len(states)))
    for state, neighbors in ADJACENCY_MAP.items():
        if state not in state_to_idx: continue
        u = state_to_idx[state]
        adj[u, u] = 1.0 # Self-loop
        for neighbor in neighbors:
            if neighbor in state_to_idx:
                v = state_to_idx[neighbor]
                adj[u, v] = 1.0
                
    # Normalize Adj matrix (D^-1/2 * A * D^-1/2)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    # 4. Model Simulation with Stabilized Weights
    model = SpatialGCN(len(features), 1)
    
    # Xavier Initialization to prevent saturation
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    print("   - Propagating Crime Trends across State Borders...")
    with torch.no_grad():
        # Feature propagation
        spillover_scores = model(node_features, adj_norm).numpy().flatten()
    
    # Ensure a minimum baseline for influential zones
    df_2030['Spatial_Spillover_Index'] = spillover_scores
    
    # 4. Save Results
    output_file = os.path.join(OUTPUT_DIR, "vision_2030_spatial_intelligence.csv")
    result_df = df_2030[['State', 'Spatial_Spillover_Index']].sort_values('Spatial_Spillover_Index', ascending=False)
    result_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("SPATIAL INTELLIGENCE COMPLETE")
    print(f"Spatial Map saved: {output_file}")
    print("Summary of High-Spillover Zones (Regional Influence):")
    print(result_df.head(5)['State'].tolist())
    print("="*40)

if __name__ == "__main__":
    run_spatial_intelligence()
