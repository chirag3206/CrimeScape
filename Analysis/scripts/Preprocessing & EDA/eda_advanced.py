import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -------- SETUP PATHS --------
INPUT_FILE = r"D:\College\Projects\PRJ-3\Crimes\clean_crime_dataset.csv"
PLOTS_DIR = r"D:\College\Projects\PRJ-3\Crimes\Analysis\results\plots\advanced"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading dataset...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please run the cleaning script first.")
    exit()

# Filter for Incidents for consistent scale
df_inc = df[df['Metric'] == 'Incidents']

def save_plot(name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved: {path}")
    plt.close()

# Set global style
sns.set_theme(style="white", palette="muted")

# -------- 1. CROSS-CATEGORY STATE HEATMAP --------
print("Generating Category-State Intensity Heatmap...")
state_cat_pivot = df_inc.pivot_table(index='State', columns='Category', values='Value', aggfunc='sum').fillna(0)
# Normalize to see which categories a state dominates in (column-wise max as 1)
state_cat_norm = state_cat_pivot.div(state_cat_pivot.max(axis=0), axis=1)

plt.figure(figsize=(12, 14))
sns.heatmap(state_cat_norm, annot=False, cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Scaled Intensity (0 to 1)'})
plt.title("Crime Intensity Hotspots: State vs Category", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Crime Category", fontsize=14)
plt.ylabel("State", fontsize=14)
save_plot("heatmap_state_vs_category")

# -------- 2. TEMPORAL STATE HEATMAPS (PER CATEGORY) --------
print("Generating Temporal State Heatmaps...")
for cat in df_inc['Category'].unique():
    subset = df_inc[df_inc['Category'] == cat]
    year_state_pivot = subset.pivot_table(index='State', columns='Year', values='Value', aggfunc='sum').fillna(0)
    
    plt.figure(figsize=(10, 14))
    sns.heatmap(year_state_pivot, annot=False, cmap="YlOrRd", linewidths=.1, cbar_kws={'label': 'Incident Count'})
    plt.title(f"Temporal Crime Growth: {cat}\n(2019-2023 Heatmap)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("State", fontsize=13)
    save_plot(f"heatmap_temporal_{cat.lower().replace(' ', '_')}")

# -------- 3. IPC VS SSL SHARE --------
print("Generating IPC vs SSL Analysis...")
# Combine Category and Law for visualization
law_share = df_inc.groupby(['Category', 'Law'])['Value'].sum().unstack().fillna(0)
# Convert to percentages for normalized comparison
law_share_pct = law_share.div(law_share.sum(axis=1), axis=0) * 100

ax = law_share_pct.plot(kind='bar', stacked=True, figsize=(10, 7), color=['#6f42c1', '#007bff', '#28a745'])
plt.title("Distribution of IPC vs SSL Crimes by Category", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Percentage Share (%)", fontsize=12)
plt.xlabel("Crime Category", fontsize=12)
plt.legend(title="Law Type", frameon=True, shadow=True)
plt.xticks(rotation=45)
# Add percentage labels
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    if height > 5: # Only label if big enough
        x, y = p.get_xy() 
        ax.text(x+width/2, y+height/2, f'{height:.1f}%', color='white', ha='center', fontweight='bold')
save_plot("law_type_share_by_category")

# -------- 4. OUTLIER ANALYSIS (BOXENPLOTS) --------
print("Generating Outlier Analysis...")
plt.figure(figsize=(14, 8))
# Boxenplots are better for large datasets than standard boxplots
sns.boxenplot(data=df_inc, x='Category', y='Value', hue='Category', palette="viridis", legend=False)
plt.yscale('log') # Log scale because crime counts vary by orders of magnitude
plt.title("Crime Count Distribution and Outliers (Log Scale)", fontsize=17, fontweight='bold', pad=20)
plt.ylabel("Reported Incidents (Log Scale)", fontsize=13)
plt.xlabel("Category", fontsize=13)
plt.grid(True, which="both", ls="-", alpha=0.3)
save_plot("outlier_distribution_log")

# -------- 5. CRIME COMPOSITION (DONUT CHARTS) --------
print("Generating Crime Composition Donut Charts...")
for cat in df_inc['Category'].unique():
    subset = df_inc[df_inc['Category'] == cat]
    # Sum incidents by Crime type
    crime_totals = subset.groupby('Crime')['Value'].sum().sort_values(ascending=False)
    
    if len(crime_totals) > 5:
        top_5 = crime_totals.head(5)
        others_val = crime_totals.iloc[5:].sum()
        plot_data = pd.concat([top_5, pd.Series({'Others': others_val})])
    else:
        plot_data = crime_totals
    
    plt.figure(figsize=(10, 10))
    # Custom palette
    colors = sns.color_palette("muted")
    
    wedges, texts, autotexts = plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', 
                                      startangle=140, pctdistance=0.85, colors=colors,
                                      textprops={'fontsize': 12})
    
    # Draw center circle to make it a donut
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.setp(autotexts, size=11, weight="bold", color="black")
    plt.title(f"Internal Composition: {cat} Crimes", fontsize=18, fontweight='bold', pad=25)
    plt.axis('equal') 
    save_plot(f"donut_composition_{cat.lower().replace(' ', '_')}")

print("\n" + "="*40)
print("ADVANCED EDA COMPLETE")
print(f"New visualizations saved in: {PLOTS_DIR}")
print("="*40)
