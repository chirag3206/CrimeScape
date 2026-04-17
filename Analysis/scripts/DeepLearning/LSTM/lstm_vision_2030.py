import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Paths
DATA_PATH = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\tables\DL\lstm_temporal_data.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\forecasts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- LSTM Architecture ---
class CrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CrimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def run_vision_2030():
    print("🧠 Initializing Reality-2030 LSTM Forecaster...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    states = df['State'].unique()
    features = [c for c in df.columns if c not in ['State', 'Year']]
    
    # 2. Scaling
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    # 3. Parameters
    sequence_length = 3  # We use 3 years to predict the 4th
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 200
    
    final_forecasts = []

    # 4. Training (One model per state for maximum precision in recursive mode)
    for state in states:
        state_data = df_scaled[df_scaled['State'] == state][features].values
        
        # Prepare sequences for training
        X, y = [], []
        for i in range(len(state_data) - sequence_length):
            X.append(state_data[i:i+sequence_length])
            y.append(state_data[i+sequence_length])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        model = CrimeLSTM(len(features), hidden_size, num_layers, len(features))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        print(f"   - Training Vision-Model for {state}...")
        for epoch in range(num_epochs):
            outputs = model(X)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 5. Recursive Forecasting (2024 to 2030)
        # Start with the last known sequence (2021, 2022, 2023)
        current_sequence = state_data[-sequence_length:]
        state_forecasts = []

        for year in range(2024, 2031):
            input_seq = torch.tensor(current_sequence.reshape(1, sequence_length, -1), dtype=torch.float32)
            with torch.no_grad():
                pred = model(input_seq).numpy()
            
            state_forecasts.append({'State': state, 'Year': year, **dict(zip(features, pred[0]))})
            
            # Update sequence (remove oldest, add newest prediction)
            current_sequence = np.append(current_sequence[1:], pred, axis=0)

        final_forecasts.extend(state_forecasts)

    # 6. Save and Invert Scaling
    forecast_df = pd.DataFrame(final_forecasts)
    forecast_df[features] = scaler.inverse_transform(forecast_df[features])
    
    # Clip negative values (as crime cannot be negative)
    forecast_df[features] = forecast_df[features].clip(lower=0)
    
    output_file = os.path.join(OUTPUT_DIR, "forecast_2030.csv")
    forecast_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("🎯 VISION 2030 FORECAST COMPLETE")
    print(f"Data saved: {output_file}")
    print("="*40)

if __name__ == "__main__":
    run_vision_2030()
