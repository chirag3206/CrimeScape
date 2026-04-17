import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Paths
HISTORICAL_DATA = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\tables\DL\lstm_temporal_data.csv"
FORECAST_2030 = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\forecasts\forecast_2030.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\embeddings"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Autoencoder Architecture ---
class StateEncoder(nn.Module):
    def __init__(self, input_size):
        super(StateEncoder, self).__init__()
        # Encoder: 500 -> 128 -> 64 -> 16 (Fingerprint)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16) 
        )
        # Decoder: 16 -> 64 -> 128 -> 500
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid() # Features are scaled 0-1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def run_state_profiling():
    print("🧬 Initializing Deep State Profiler (Autoencoder)...")
    
    # 1. Load Data
    hist_df = pd.read_csv(HISTORICAL_DATA)
    fore_df = pd.read_csv(FORECAST_2030)
    
    features = [c for c in hist_df.columns if c not in ['State', 'Year']]
    
    # 2. Preprocessing
    scaler = MinMaxScaler()
    # Combine all data to learn a global feature space
    combined_data = pd.concat([hist_df[features], fore_df[features]])
    X_scaled = scaler.fit_transform(combined_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # 3. Training
    model = StateEncoder(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("   - Learning the Sociological Latent Space...")
    for epoch in range(200):
        latent, reconstructed = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 4. Extract Fingerprints (Embeddings)
    with torch.no_grad():
        embeddings, _ = model(X_tensor)
        embeddings = embeddings.numpy()
    
    # 5. Create Comparison Map (2023 vs 2030)
    # We'll take 2023 and 2030 to see the shifts
    results_df = pd.concat([
        hist_df[['State', 'Year']],
        fore_df[['State', 'Year']]
    ]).reset_index(drop=True)
    
    for i in range(16):
        results_df[f'Dim_{i+1}'] = embeddings[:, i]
        
    output_file = os.path.join(OUTPUT_DIR, "state_fingerprints_2030.csv")
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("🧬 STATE PROFILING COMPLETE")
    print(f"Embeddings saved: {output_file}")
    print("The model has compressed 500 variables into 16 unique 'DNA' markers per state.")
    print("="*40)

if __name__ == "__main__":
    run_state_profiling()
