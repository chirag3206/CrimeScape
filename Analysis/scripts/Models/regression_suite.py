import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
INPUT_FILE = os.path.join(BASE_DIR, r"Analysis\results\tables\ml_ready_pivoted_rates.csv")
MODELS_ROOT = os.path.join(BASE_DIR, "Models")
PLOTS_DIR = os.path.join(BASE_DIR, r"Analysis\results\plots\regression")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading pre-processed dataset for Regression Suite...")
df = pd.read_csv(INPUT_FILE, index_col='State').fillna(0)

# Domains to analyze
DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def perform_regression_suite(domain_name):
    print(f"\n--- REGRESSION SUITE: {domain_name} ---")
    
    # Filter columns for this domain
    domain_cols = [col for col in df.columns if col.startswith(f"{domain_name}_")]
    if not domain_cols:
        print(f"Warning: No data for {domain_name}.")
        return

    # Target (y): Domain intensity (Sum of columns)
    # Features (X): Individual crime types
    X = df[domain_cols]
    y = X.sum(axis=1)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Performance Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R-squared": r2}
        
        # Save Model
        model_path = os.path.join(MODELS_ROOT, domain_name, "Regression")
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model, os.path.join(model_path, f"{name.lower()}_model.joblib"))
        
        print(f"{name} - R-squared Accuracy: {r2:.4f}, MSE: {mse:.4e}")

    # --- VISUALIZE COMPARISON ---
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Model', y='R-squared', hue='Model', palette="Set2", legend=False)
    plt.title(f"Predictive Fit Comparison: {domain_name} (R-squared Accuracy)", fontsize=14, fontweight='bold')
    plt.ylabel("R-squared (Accuracy of Fit)")
    plt.ylim(0, 1.1)
    
    plot_fn = f"regression_comparison_{domain_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_fn), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Regression Suite complete. Performance charts and models saved in: {model_path}")

for domain in DOMAINS:
    perform_regression_suite(domain)

print("\n" + "="*50)
print("MULTI-DOMAIN REGRESSION SUITE COMPLETE")
print("="*50)
