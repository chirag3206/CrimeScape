import pandas as pd
import numpy as np

# -------- LOAD DATA --------
print("Loading master dataset...")
df = pd.read_csv(r"D:\College\Projects\PRJ-3\Crimes\master_crime_dataset.csv", low_memory=False)

# -------- STEP 1: ROBUST FEATURE SPLIT --------
# Detect metrics (Incidents, Victims, Rate) based on suffixes or default to Incidents
def split_feature(feature):
    if str(feature).endswith("_I"):
        return feature[:-2], "Incidents"
    elif str(feature).endswith("_V"):
        return feature[:-2], "Victims"
    elif str(feature).endswith("_R"):
        return feature[:-2], "Rate"
    else:
        # Descriptive columns or totals (treat as Incidents/Count)
        return feature, "Incidents"

print("Splitting features into Crime and Metric...")
temp_split = df["Feature"].apply(split_feature)
df["Crime"] = [x[0] for x in temp_split]
df["Metric"] = [x[1] for x in temp_split]

# -------- STEP 2: FILTER METRICS --------
# Keep both Incidents and Victims for full analysis/EDA. Drop Rate as it's a derived metric.
df = df[df["Metric"].isin(["Incidents", "Victims"])]

# -------- STEP 3: STANDARDIZE STATE NAMES --------
# Mapping abbreviations to full descriptive names for better visualizations
state_name_map = {
    "A&N Islands": "Andaman & Nicobar Islands",
    "D&N Haveli and Daman & Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "Delhi": "Delhi (UT)",
}

df["State"] = df["State"].replace(state_name_map)
df["State"] = df["State"].str.strip()

# -------- STEP 4: CLEAN AND CONVERT VALUES --------
# Ensure values are numeric and remove any remaining nulls
df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
df = df.dropna(subset=["Value"])

# -------- STEP 5: REMOVE UNNECESSARY COLUMNS --------
# We keep Metric because some ML models might want to distinguish Incidents/Victims
df = df.drop(columns=["Feature", "Source_File"], errors='ignore')

# -------- STEP 6: FINAL POLISH --------
# Remove duplicates and sort for consistency
df = df.drop_duplicates()
df = df.sort_values(["Year", "Category", "State", "Crime", "Metric"])

# -------- SAVE CLEAN DATA --------
output_path = r"D:\College\Projects\PRJ-3\Crimes\clean_crime_dataset.csv"
df.to_csv(output_path, index=False)

print("\n" + "="*40)
print("CLEAN DATASET READY FOR ML")
print(f"Saved at: {output_path}")
print(f"Total rows: {len(df)}")
print(f"Unique Crimes detected: {df['Crime'].nunique()}")
print(f"Metrics preserved: {df['Metric'].unique()}")
print("="*40)
print(df.head())