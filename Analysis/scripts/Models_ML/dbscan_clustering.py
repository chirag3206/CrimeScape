import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
INPUT_FILE = os.path.join(BASE_DIR, r"Analysis\results\tables\ml_ready_pivoted_rates.csv")
MODELS_ROOT = os.path.join(BASE_DIR, "Models")
PLOTS_DIR = os.path.join(BASE_DIR, r"Analysis\results\plots\clustering")
# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading pre-processed dataset for clustering...")
try:
    df = pd.read_csv(INPUT_FILE, index_col='State')
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please run the advanced EDA script first to generate ml_ready_pivoted_rates.csv.")
    exit()

# Domains to analyze separately
DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def perform_clustering(domain_name):
    print(f"\n--- CLUSTERING FOR: {domain_name} ---")
    
    # Filter columns specifically for this domain
    domain_cols = [col for col in df.columns if col.startswith(f"{domain_name}_")]
    if not domain_cols:
        print(f"Warning: No data columns found for domain {domain_name}.")
        return
        
    X = df[domain_cols].fillna(0)
    
    # 1. Standardize the data (Vital for distance-based clustering like DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Fit DBSCAN
    # Relaxing eps to 2.0 because of the small sample size (36 states)
    db = DBSCAN(eps=2.0, min_samples=2)
    labels = db.fit_predict(X_scaled)
    
    # 3. Evaluation Matrix (Accuracy/Quality Metric)
    # Silhouette Score measures how similar an object is to its own cluster compared to others
    unique_labels = set(labels) - {-1} # Exclude noise for silhouette
    if len(unique_labels) > 1:
        sil_score = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score (Accuracy/Quality): {sil_score:.4f}")
    else:
        print("Model generated only 1 cluster (+ noise). Silhouette score cannot be computed.")
        
    print(f"Noise points detected (-1): {list(labels).count(-1)}")
    print(f"Number of clusters: {len(unique_labels)}")

    # 4. Save Trained Model & Scaler
    model_path = os.path.join(MODELS_ROOT, domain_name, "Clustering")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(db, os.path.join(model_path, "dbscan_model.joblib"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.joblib"))
    
    # 5. Visualizing Clusters with PCA (Dimensionality Reduction to 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    # Define color palette, handling noise (-1) specially if needed
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis", s=150, alpha=0.9, style=labels)
    plt.title(f"National State Archetypes: {domain_name}\n(DBSCAN Grouping)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f"PCA Dimension 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PCA Dimension 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(title="Cluster Label (-1 = Noise)", loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plot_fn = f"dbscan_{domain_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_fn), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 6. Save Membership Table
    results = pd.DataFrame({"Cluster_Label": labels}, index=X.index)
    results = results.sort_values("Cluster_Label")
    results.to_csv(os.path.join(model_path, "cluster_assignments.csv"))
    print(f"Models and results saved in: {model_path}")

# Run for each of the 4 requested domains
for domain in DOMAINS:
    perform_clustering(domain)

print("\n" + "="*50)
print("DBSCAN MULTI-DOMAIN ANALYSIS COMPLETE")
print(f"Visual Gallery available in: {PLOTS_DIR}")
print("="*50)
