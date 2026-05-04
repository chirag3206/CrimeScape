import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os

# Paths
DATA_PATH = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\tables\DL\lstm_temporal_data.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\forecasts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- National LSTM Architecture ---
class NationalCrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(NationalCrimeLSTM, self).__init__()
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

def run_national_vision_2030():
    print("Initializing Rolling National Vision-2030 LSTM...")
    
    # 1. Load and Aggregate Data
    df_raw = pd.read_csv(DATA_PATH)
    features = [c for c in df_raw.columns if c not in ['State', 'Year']]
    
    print("   - Aggregating India-Total Crime Data (2019-2023)...")
    df_national = df_raw.groupby('Year')[features].sum().reset_index().sort_values('Year')
    
    # 2. Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_national[features].values)
    
    # 3. Model Parameters
    seq_len = 2 # Best for short historical window
    hidden_size = 32
    num_layers = 1
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    model = NationalCrimeLSTM(len(features), hidden_size, num_layers, len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 4. Training (National Training)
    print("   - Training Unified National Forecasting Engine...")
    for epoch in range(400):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # 5. Rolling Forecast (2024 to 2030)
    print("   - Generating Rolling Forecast till 2030...")
    current_sequence = data_scaled[-seq_len:]
    final_forecasts = []

    for year in range(2024, 2031):
        input_seq = torch.tensor(current_sequence.reshape(1, seq_len, -1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(input_seq).numpy()
        
        # Invert scaling
        pred_unscaled = scaler.inverse_transform(pred).flatten()
        final_forecasts.append({'State': 'India_Total', 'Year': year, **dict(zip(features, pred_unscaled))})
        
        # Prepare for next rolling step
        current_sequence = np.append(current_sequence[1:], pred, axis=0)

    # 6. Save results
    forecast_df = pd.DataFrame(final_forecasts)
    # Ensure no negative predictions
    numerical_cols = [c for c in forecast_df.columns if c not in ['State', 'Year']]
    forecast_df[numerical_cols] = forecast_df[numerical_cols].clip(lower=0)
    
    output_file = os.path.join(OUTPUT_DIR, "national_forecast_2030.csv")
    forecast_df.to_csv(output_file, index=False)
    
    print("\n" + "="*40)
    print("NATIONAL VISION 2030 COMPLETE")
    print(f"Prophetic Data saved: {output_file}")
    print("="*40)

if __name__ == "__main__":
    run_national_vision_2030()
