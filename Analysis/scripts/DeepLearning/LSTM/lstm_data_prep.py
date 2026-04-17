import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Paths
INPUT_PATH = r"d:\College\Projects\PRJ-3\Crimes\master_crime_dataset.csv"
OUTPUT_DIR = r"d:\College\Projects\PRJ-3\Crimes\Analysis\results\tables\DL"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lstm_temporal_data.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def prepare_lstm_data():
    print("🚀 Starting LSTM Data Preparation...")
    
    # Load Master Dataset
    df = pd.read_csv(INPUT_PATH)
    
    # Create a unique Feature identifier (Category + Feature Name)
    df['Full_Feature'] = df['Category'] + "_" + df['Feature']
    
    # Pivot the data: Rows as (State, Year), Columns as Full_Feature
    # We use 'Value' as the data
    print("📊 Pivoting dataset (this may take a moment)...")
    pivoted = df.pivot_table(
        index=['State', 'Year'],
        columns='Full_Feature',
        values='Value',
        aggfunc='mean'
    ).reset_index()
    
    # Handle Missing Values
    # Since not every state has every crime reported every year, we fill NaNs
    # 1. Fill with 0 for rare crimes
    # 2. Fill with state-wise mean for continuity if 0 doesn't make sense
    pivoted = pivoted.fillna(0)
    
    # Sort by State and Year for temporal consistency
    pivoted = pivoted.sort_values(['State', 'Year'])
    
    # Save the temporal dataset
    pivoted.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ LSTM Temporal Data saved to: {OUTPUT_FILE}")
    print(f"📈 Total rows: {len(pivoted)}")
    print(f"📉 Total features: {len(pivoted.columns) - 2}")
    
    return pivoted

if __name__ == "__main__":
    prepare_lstm_data()
