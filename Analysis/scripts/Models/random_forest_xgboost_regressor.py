import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
INPUT_FILE = os.path.join(BASE_DIR, r"Analysis\results\tables\ml_ready_pivoted_rates.csv")
MODELS_ROOT = os.path.join(BASE_DIR, "Models")
PLOTS_DIR = os.path.join(BASE_DIR, r"Analysis\results\plots\importance")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading pre-processed dataset for Regression/Importance...")
df = pd.read_csv(INPUT_FILE, index_col='State').fillna(0)

DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def perform_importance_analysis(domain_name):
    print(f"\n--- FEATURE IMPORTANCE & REGRESSION: {domain_name} ---")
    
    # Filter columns for THIS domain
    domain_cols = [col for col in df.columns if col.startswith(f"{domain_name}_")]
    if not domain_cols:
        print(f"Warning: No data for {domain_name}.")
        return

    # Feature Matrix (X): Individual crime types
    # Target (y): Total Normalized Rate for that domain
    X = df[domain_cols].copy()
    y = X.sum(axis=1) # Target is the domain-wide intensity
    
    # Sanitize feature names for XGBoost (disallows [, ], <)
    X.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X.columns]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- MODEL 1: RANDOM FOREST ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # --- MODEL 2: XGBOOST ---
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    # Accuracy Metric: R-squared (represents goodness of fit)
    print(f"Random Forest - R-squared Accuracy: {r2_rf:.4f}")
    print(f"XGBoost - R-squared Accuracy: {r2_xgb:.4f}")
    
    # Save Best Model (XGBoost usually better)
    model_path = os.path.join(MODELS_ROOT, domain_name, "Regression")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(xgb, os.path.join(model_path, "xgboost_regressor_model.joblib"))
    joblib.dump(rf, os.path.join(model_path, "random_forest_model.joblib"))
    
    # --- VISUALIZE FEATURE IMPORTANCE ---
    # Using Random Forest for stable importance scores
    importances = rf.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette="rocket", legend=False)
    plt.title(f"Dominant Crime Drivers (Feature Importance): {domain_name}", fontsize=15, fontweight='bold')
    plt.xlabel("Importance Score")
    
    plot_fn = f"importance_{domain_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_fn), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Analysis complete. Visualizations and models saved in: {model_path}")

for domain in DOMAINS:
    perform_importance_analysis(domain)

print("\n" + "="*50)
print("ENSEMBLE REGRESSION & IMPORTANCE COMPLETE")
print("="*50)
