import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
ML_BASE = os.path.join(BASE_DIR, r"Internal_Validation\ML_Validation")
DATA_DIR = os.path.join(ML_BASE, "data")
PLOTS_DIR = os.path.join(ML_BASE, "plots")
REPORTS_DIR = os.path.join(ML_BASE, "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MASTER_LOG = os.path.join(REPORTS_DIR, "ml_confirmation_2022.log")

def log_msg(msg):
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# Initialize Log
with open(MASTER_LOG, "w", encoding="utf-8") as f:
    f.write("=== ML CONFIRMATION REPORT (TARGET: 2022) ===\n")
    f.write("Training: 2019-2021 | Validation: 2022 Actuals\n")
    f.write("============================================\n\n")

# -------- LOAD DATA --------
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_19_21.csv"), index_col='State')
df_actual = pd.read_csv(os.path.join(DATA_DIR, "actual_2022.csv"), index_col='State')

DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def run_confirmation():
    summary_results = []
    
    for domain in DOMAINS:
        log_msg(f"\n>> Validating Domain: {domain}")
        
        # Features & Target
        cols = [c for c in df_train.columns if c.startswith(f"{domain}_")]
        X_train = df_train[cols].copy()
        y_train = X_train.sum(axis=1)
        
        X_test = df_actual[cols].copy()
        y_test = X_test.sum(axis=1)
        
        # Sanitize for XGBoost
        X_train.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_train.columns]
        X_test.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_test.columns]
        
        # 1. REGRESSION (XGBoost)
        reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        
        log_msg(f"   [Regressor] R2: {r2:.4f} | MAPE: {mape:.2f}%")
        
        # Plot 1: Actual vs Predicted Bar
        plt.figure(figsize=(12, 6))
        x = np.arange(len(y_test))
        plt.bar(x - 0.2, y_test, 0.4, label='Actual 2022', color='#2ecc71')
        plt.bar(x + 0.2, y_pred, 0.4, label='Predicted', color='#3498db')
        plt.xticks(x, y_test.index, rotation=90)
        plt.title(f"{domain}: Actual vs Predicted (2022)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"bar_2022_{domain.lower().replace(' ', '_')}.png"))
        plt.close()
        
        # 2. CLASSIFICATION (Lasso Logistic)
        threshold = y_train.median()
        y_train_clf = (y_train > threshold).astype(int)
        y_test_clf = (y_test > threshold).astype(int)
        
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
        clf.fit(X_train, y_train_clf)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred_clf = clf.predict(X_test)
        
        acc = accuracy_score(y_test_clf, y_pred_clf)
        f1 = f1_score(y_test_clf, y_pred_clf)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
        roc_auc = auc(fpr, tpr)
        
        log_msg(f"   [Classifier] Acc: {acc:.2f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
        
        # Plot 2: ROC Curve
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {domain} (2022)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(PLOTS_DIR, f"roc_2022_{domain.lower().replace(' ', '_')}.png"))
        plt.close()

        summary_results.append({
            "Domain": domain, "R2": r2, "MAPE": mape, "Accuracy": acc, "AUC": roc_auc
        })
        
    # Summary Table
    sdf = pd.DataFrame(summary_results)
    log_msg("\n=== FINAL ML CONFIRMATION TABLE ===")
    log_msg(sdf.to_string(index=False))
    sdf.to_csv(os.path.join(REPORTS_DIR, "ml_metrics_summary.csv"), index=False)

if __name__ == "__main__":
    run_confirmation()
    print("\nML Confirmation Complete. See Internal_Validation/ML_Validation/")
