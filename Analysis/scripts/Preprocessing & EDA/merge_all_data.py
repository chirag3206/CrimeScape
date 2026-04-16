import os
import pandas as pd
import re

# -------- ROOT PATH --------
BASE_PATH = r"D:\College\Projects\PRJ-3\Crimes\CSV"
OUTPUT_FILE = r"D:\College\Projects\PRJ-3\Crimes\master_crime_dataset.csv"

# -------- STATE NORMALIZATION MAP --------
STATE_MAP = {
    "A & N Islands": "A&N Islands",
    "A&N Islands": "A&N Islands",
    "Delhi UT": "Delhi",
    "Delhi": "Delhi",
    "D&N Haveli and Daman & Diu": "D&N Haveli and Daman & Diu",
    "D&N Haveli": "D&N Haveli and Daman & Diu",
    "Daman & Diu": "D&N Haveli and Daman & Diu",
    "DNH and Daman & Diu": "D&N Haveli and Daman & Diu",
    "D&N Haveli and Daman": "D&N Haveli and Daman & Diu",
    "Dadra & Nagar Haveli and Daman & Diu": "D&N Haveli and Daman & Diu",
    "DNH and Daman": "D&N Haveli and Daman & Diu",
    "Dadra & Nagar Haveli": "D&N Haveli and Daman & Diu",
    "Arunachal": "Arunachal Pradesh",
    "D&N Haveli and": "D&N Haveli and Daman & Diu",
    "Jammu & Kashmir": "Jammu & Kashmir",
    "Odisha": "Odisha",
    "Orissa": "Odisha",
}

def clean_state(name):
    if pd.isna(name): return None
    name = str(name).strip().replace("*", "")
    
    # Remove trailing numbers or footnotes (e.g., "Assam 1", "Bihar (2)")
    name = re.sub(r'\d+$', '', name)
    name = re.sub(r'\(\d+\)$', '', name).strip()
    
    if not name: return None

    # Check for total rows or junk - skip these to avoid double counting
    upper_name = name.upper()
    junk_keywords = ["TOTAL", "ALL INDIA", "STATES/UTS", "R/W G (SEC", "(SEC"]
    if any(keyword in upper_name for keyword in junk_keywords):
        return None
    
    # Filter out single character names or symbols
    if len(name) < 2 or name == "&":
        return None
        
    return STATE_MAP.get(name, name)

# -------- STORAGE --------
all_data = []

print("Starting master dataset creation...")

# -------- WALK THROUGH DIRECTORY --------
for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if not file.endswith(".csv"):
            continue
            
        file_path = os.path.join(root, file)
        
        # Extract metadata from path
        rel_path = os.path.relpath(file_path, BASE_PATH)
        parts = rel_path.split(os.sep)
        
        if len(parts) < 2:
            continue
            
        category = parts[0]
        year = parts[1]
        
        # Determine Law Type (IPC, SSL, etc.)
        if len(parts) >= 4:
            law_type = parts[2]
        else:
            law_type = "Other"

        try:
            # Use low_memory=False to avoid DtypeWarnings
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            
            # Find the State column
            state_col = next((c for c in df.columns if "STATE" in c.upper() or c.upper() == "UT" or c.upper() == "UT/STATE"), None)
            if not state_col:
                # If no clear state column, skip
                continue
            
            # Rename to standard "State"
            df = df.rename(columns={state_col: "State"})
            
            # -------- REMOVE ID/SERIAL COLUMNS --------
            # These vary between SL, SNo, S.No, Index, etc.
            id_patterns = ["SL", "SNO", "S.NO", "INDEX", "S NO", "SR. NO.", "S. NO.", "SN"]
            id_cols_to_drop = [c for c in df.columns if c.upper() in id_patterns]
            df = df.drop(columns=id_cols_to_drop)
            
            # -------- CLEAN STATE NAMES --------
            df["State"] = df["State"].apply(clean_state)
            df = df.dropna(subset=["State"])
            
            if df.empty:
                continue

            # Ensure all column names are strings
            df.columns = [str(c) for c in df.columns]

            # -------- CONVERT TO LONG FORMAT --------
            # Melt everything except 'State'
            df_long = df.melt(id_vars=["State"], var_name="Feature", value_name="Value")
            
            # -------- ADD METADATA --------
            try:
                df_long["Year"] = int(year)
            except:
                df_long["Year"] = 0 # Fallback for non-numeric year folders if any
                
            df_long["Category"] = category
            df_long["Law"] = law_type
            df_long["Source_File"] = file
            
            all_data.append(df_long)
            
        except Exception as e:
            print(f"❌ Error in file: {file_path}")
            print(f"   Reason: {e}")

# -------- FINAL DATAFRAME --------
if not all_data:
    print("⚠️ No valid data found to merge!")
else:
    print("Concatenating dataframes...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Final cleanup of Value
    def clean_value(val):
        if pd.isna(val): return val
        if isinstance(val, str):
            # Remove commas and non-numeric junk except period
            val = val.replace(",", "").strip()
            # If empty after stripping
            if not val: return None
        return val

    print("Cleaning numeric values...")
    final_df["Value"] = final_df["Value"].apply(clean_value)
    final_df["Value"] = pd.to_numeric(final_df["Value"], errors='coerce')
    
    # Drop rows with NaN values (either they were junk or total rows we filtered)
    final_df = final_df.dropna(subset=["Value"])
    
    # Final sorting for beauty
    final_df = final_df.sort_values(["Year", "Category", "State"])

    # -------- SAVE --------
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("MASTER DATASET CREATED SUCCESSFULLY")
    print(f"Saved at: {OUTPUT_FILE}")
    print(f"Total records (features): {len(final_df)}")
    print(f"Unique States/UTs: {final_df['State'].nunique()}")
    print("="*40)