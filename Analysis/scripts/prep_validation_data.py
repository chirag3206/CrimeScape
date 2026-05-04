import pandas as pd
import numpy as np
import os

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
MASTER_DATA = os.path.join(BASE_DIR, "master_crime_dataset.csv")
POP_DATA = os.path.join(BASE_DIR, r"Analysis\data\india_pop_2021.csv")

# NEW OUTPUT LOCATIONS
ML_DATA_DIR = os.path.join(BASE_DIR, r"Internal_Validation\ML_Validation\data")
DL_DATA_DIR = os.path.join(BASE_DIR, r"Internal_Validation\DL_Validation\data")

os.makedirs(ML_DATA_DIR, exist_ok=True)
os.makedirs(DL_DATA_DIR, exist_ok=True)

# -------- AGGREGATE COLUMNS TO FILTER --------
AGGREGATE_COLUMNS = [
    'Kidnap_Total', 'Selling_Total', 'Buying_Total', 'Total', 'POCSO_Total', 
    'Sec4_6_Total', 'Sec8_10_Total', 'Sec12_Total', 'Sec14_15_Total', 
    'Sec377_Total', 'Sec17_22_Total', 'JJ_Total', 'ITP_Total', 'Cyber_Total', 
    'Total_SLL', 'Total_All', 'Total Persons', 'Total_Apprehended', 
    'Total_Above18', 'Kidnap_Total_Above18', 'GrandTotal', 'Total_Crime', 'Total_IPC'
]

def prepare_split_validation_data():
    print("Starting Split-Target Data Preparation...")
    
    # 1. Load Data
    df = pd.read_csv(MASTER_DATA)
    df_pop = pd.read_csv(POP_DATA)
    
    # 2. Filtering
    print("   - Filtering aggregate rows...")
    df = df[~df["Feature"].isin(AGGREGATE_COLUMNS)]
    
    # 3. Normalization (Per 100k)
    print("   - Normalizing by population...")
    df = df.merge(df_pop, on="State", how="left")
    df["Crime_Rate"] = df["Value"] / df["Population_2021_Lakhs"]
    
    # helper to pivot a year subset
    def pivot_subset(subset):
        pivoted = subset.pivot_table(
            index='State', 
            columns=['Category', 'Feature'], 
            values='Crime_Rate', 
            aggfunc='mean'
        ).fillna(0)
        pivoted.columns = [f"{c[0]}_{c[1]}" for c in pivoted.columns]
        return pivoted

    # 4. Create Splits
    print("   - Processing Training Split (2019-2021)...")
    train_df = df[df["Year"].isin([2019, 2020, 2021])].copy()
    train_pivot = pivot_subset(train_df)
    
    print("   - Processing ML Confirmation Target (2022)...")
    ml_target_df = df[df["Year"] == 2022].copy()
    ml_target_pivot = pivot_subset(ml_target_df)
    
    print("   - Processing DL Confirmation Target (2023)...")
    dl_target_df = df[df["Year"] == 2023].copy()
    dl_target_pivot = pivot_subset(dl_target_df)
    
    # 5. Synchronize Columns (Ensure targets match training features exactly)
    all_train_cols = train_pivot.columns.tolist()
    ml_target_pivot = ml_target_pivot.reindex(columns=all_train_cols, fill_value=0)
    dl_target_pivot = dl_target_pivot.reindex(columns=all_train_cols, fill_value=0)
    
    # 6. Save Outputs
    # Training goes to both for ease of access
    train_pivot.to_csv(os.path.join(ML_DATA_DIR, "train_19_21.csv"))
    train_pivot.to_csv(os.path.join(DL_DATA_DIR, "train_19_21.csv"))
    
    # Ground Truths
    ml_target_pivot.to_csv(os.path.join(ML_DATA_DIR, "actual_2022.csv"))
    dl_target_pivot.to_csv(os.path.join(DL_DATA_DIR, "actual_2023.csv"))
    
    print("\n" + "="*40)
    print(" DONE: SPLIT-TARGET DATASETS READY")
    print(f" ML Base : {ML_DATA_DIR}")
    print(f" DL Base : {DL_DATA_DIR}")
    print("="*40)

if __name__ == "__main__":
    prepare_split_validation_data()
