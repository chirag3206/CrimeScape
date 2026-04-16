import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------- SETUP PATHS --------
INPUT_FILE = r"D:\College\Projects\PRJ-3\Crimes\clean_crime_dataset.csv"
PLOTS_DIR = r"D:\College\Projects\PRJ-3\Crimes\Analysis\results\plots"
TABLES_DIR = r"D:\College\Projects\PRJ-3\Crimes\Analysis\results\tables"

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading dataset...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please run the cleaning script first.")
    exit()

# Set global style for premium look
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.family'] = 'sans-serif'

def save_plot(name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved: {path}")
    plt.close()

# -------- 1. TREND ANALYSIS --------
print("Analyzing yearly trends...")
trends = df.groupby(['Year', 'Category'])['Value'].sum().reset_index()
plt.figure()
sns.lineplot(data=trends, x='Year', y='Value', hue='Category', marker='o', linewidth=3)
plt.title("Crime Incidents Trend (2019-2023) by Category", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Total Incidents / Victims Count", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.xticks([2019, 2020, 2021, 2022, 2023])
plt.legend(title="Domain", title_fontsize='13', fontsize='11')
save_plot("yearly_category_trends")

# -------- 2. STATE-WISE DISTRIBUTION --------
print("Analyzing state-wise distributions...")
for cat in df['Category'].unique():
    plt.figure()
    subset = df[df['Category'] == cat]
    state_totals = subset.groupby('State')['Value'].sum().sort_values(ascending=False).head(10)
    
    sns.barplot(x=state_totals.values, y=state_totals.index, hue=state_totals.index, palette="Spectral", legend=False)
    plt.title(f"Top 10 States - {cat}\n(Accumulated 2019-2023)", fontsize=15, fontweight='bold')
    plt.xlabel("Total Count", fontsize=12)
    plt.ylabel("State", fontsize=12)
    save_plot(f"top_10_states_{cat.lower().replace(' ', '_')}")

# -------- 3. INCIDENT VS VICTIM CORRELATION --------
print("Analyzing incident vs victim correlation...")
# Pivot to get Incidents and Victims in separate columns
corr_df = df.pivot_table(index=['Year', 'Category', 'State', 'Crime'], 
                         columns='Metric', 
                         values='Value').reset_index().dropna()

if not corr_df.empty:
    plt.figure()
    sns.scatterplot(data=corr_df, x='Incidents', y='Victims', hue='Category', alpha=0.5, s=60)
    plt.title("Correlation: Incidents vs Victims per Crime Entry", fontsize=15, fontweight='bold')
    plt.xlabel("Number of Incidents", fontsize=12)
    plt.ylabel("Number of Victims", fontsize=12)
    # Adding a diagonal line for 1:1 reference
    max_val = max(corr_df['Incidents'].max(), corr_df['Victims'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', alpha=0.5)
    save_plot("incident_victim_correlation")

# -------- 4. TOP CRIME SUB-TYPES --------
print("Analyzing most common crimes...")
for cat in df['Category'].unique():
    plt.figure()
    # Filter for incidents only to see which crimes are reported most
    subset = df[(df['Category'] == cat) & (df['Metric'] == 'Incidents')]
    crime_totals = subset.groupby('Crime')['Value'].sum().sort_values(ascending=False).head(10)
    
    if not crime_totals.empty:
        sns.barplot(x=crime_totals.values, y=crime_totals.index, hue=crime_totals.index, palette="rocket", legend=False)
        plt.title(f"Most Reported Crime Types - {cat}", fontsize=15, fontweight='bold')
        plt.xlabel("Total Reported Incidents", fontsize=12)
        save_plot(f"top_crimes_{cat.lower().replace(' ', '_')}")

# -------- 5. GROWTH ANALYSIS (YoY) --------
print("Analyzing growth rates...")
# Calculate percentage change for each category
pivot_trends = trends.pivot(index='Year', columns='Category', values='Value')
growth = pivot_trends.pct_change() * 100
growth = growth.dropna().reset_index().melt(id_vars='Year', var_name='Category', value_name='Growth_Rate')

plt.figure()
sns.barplot(data=growth, x='Year', y='Growth_Rate', hue='Category', palette="Set2")
plt.axhline(0, color='black', linewidth=1, alpha=0.7)
plt.title("Year-on-Year Crime Growth Rate (%)", fontsize=16, fontweight='bold')
plt.ylabel("Growth Rate (%)", fontsize=12)
plt.legend(title="Domain", loc='upper left')
save_plot("yoy_growth_rate")

# -------- SAVE SUMMARY TABLES --------
print("Saving summary tables...")
trends.to_csv(os.path.join(TABLES_DIR, "yearly_summary.csv"), index=False)
category_state_summary = df.groupby(['Category', 'State', 'Metric'])['Value'].sum().unstack().reset_index()
category_state_summary.to_csv(os.path.join(TABLES_DIR, "category_state_comparison.csv"), index=False)

print("\n" + "="*40)
print("EDA PROCESS COMPLETE")
print(f"Visualizations saved: {PLOTS_DIR}")
print(f"Data summaries saved: {TABLES_DIR}")
print("="*40)
