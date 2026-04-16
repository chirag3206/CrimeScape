import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
INPUT_FILE = os.path.join(BASE_DIR, r"Analysis\results\tables\ml_ready_pivoted_rates.csv")
MODELS_ROOT = os.path.join(BASE_DIR, "Models")
PLOTS_DIR = os.path.join(BASE_DIR, r"Analysis\results\plots\classification")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------- LOAD DATA --------
print("Loading pre-processed dataset for Classification...")
df = pd.read_csv(INPUT_FILE, index_col='State').fillna(0)

# Domains to analyze
DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def perform_classification_analysis(domain_name):
    print(f"\n--- HIGH-RISK CLASSIFICATION: {domain_name} ---")
    
    # Filter columns for this domain
    domain_cols = [col for col in df.columns if col.startswith(f"{domain_name}_")]
    if not domain_cols:
        print(f"Warning: No data for {domain_name}.")
        return

    # 1. Binary Target Creation: High-Risk (1) vs. Low-Risk (0)
    # Threshold = Median of total domain crime rate
    domain_total = df[domain_cols].sum(axis=1)
    threshold = domain_total.median()
    y = (domain_total > threshold).astype(int)
    X = df[domain_cols]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- MODEL 1: STANDARD LOGISTIC REGRESSION ---
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    
    # --- MODEL 2: LASSO LOGISTIC REGRESSION ---
    # Penalty='l1', solver='liblinear' enables Lasso feature selection
    lasso_log = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
    lasso_log.fit(X_train, y_train)
    y_pred_lasso = lasso_log.predict(X_test)
    acc_lasso = accuracy_score(y_test, y_pred_lasso)
    
    print(f"Standard Logistic Accuracy: {acc_log:.4f}")
    print(f"Lasso Logistic Accuracy: {acc_lasso:.4f}")
    
    # Save Models
    model_path = os.path.join(MODELS_ROOT, domain_name, "Classification")
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(log_reg, os.path.join(model_path, "logistic_model.joblib"))
    joblib.dump(lasso_log, os.path.join(model_path, "lasso_logistic_model.joblib"))
    
    # --- VISUALIZE CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred_lasso)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix: {domain_name} High-Risk Classifier", fontsize=15, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plot_fn = f"confusion_matrix_{domain_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_fn), bbox_inches='tight', dpi=300)
    plt.close()
    
    # --- VISUALIZE LASSO COEFFICIENTS (Feature Indicators) ---
    coeffs = pd.Series(lasso_log.coef_[0], index=X.columns).sort_values()
    # Take top 5 positive and top 5 negative indicators
    top_coeffs = pd.concat([coeffs.head(5), coeffs.tail(5)])
    
    plt.figure(figsize=(10, 8))
    top_coeffs.plot(kind='barh', color=(top_coeffs > 0).map({True: 'red', False: 'green'}))
    plt.title(f"High-Risk Indicators (Lasso Coefficients): {domain_name}", fontsize=14)
    plt.xlabel("Coefficient Weight (Lasso)")
    
    plot_fn_coef = f"risk_indicators_{domain_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(PLOTS_DIR, plot_fn_coef), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Classification complete. Results and models saved in: {model_path}")

for domain in DOMAINS:
    perform_classification_analysis(domain)

print("\n" + "="*50)
print("MULTI-DOMAIN CLASSIFICATION PHASE COMPLETE")
print("="*50)
