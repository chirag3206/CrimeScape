import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# -------- SETUP PATHS --------
BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
DL_VALIDATION_DIR = os.path.join(BASE_DIR, r"Internal_Validation\DL_Validation")
PLOTS_DIR = os.path.join(DL_VALIDATION_DIR, "plots")
REPORTS_DIR = os.path.join(DL_VALIDATION_DIR, "reports")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MASTER_LOG = os.path.join(REPORTS_DIR, "full_spectrum_dl_audit_2023.log")

def log_msg(msg):
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

with open(MASTER_LOG, "w", encoding="utf-8") as f:
    f.write("=== PURE NEURAL MULTI-DOMAIN AUDIT (SCIENTIFIC TRUTH) ===\n")
    f.write("Method: 2-Layer Bidirectional LSTM | NO CALIBRATION | NO BIAS-CORRECTION\n")
    f.write("========================================================\n\n")

# -------- MODELS --------
class NationalCrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(NationalCrimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# -------- EXECUTION --------
DOMAINS = ["Women", "Children", "Juvenile", "Human Traficking"]

def run_full_spectrum_audit():
    log_msg("Initializing Pure Neural Validation (Transparent Intelligence)...")
    DATA_PATH = os.path.join(BASE_DIR, r"Analysis\results\tables\DL\lstm_temporal_data.csv")
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c not in ['State', 'Year']]
    domain_feats = {d: [f for f in features if f.startswith(d)] for d in DOMAINS}
    
    df_nat = df.groupby('Year')[features].sum().reset_index().sort_values('Year')
    sc_minmax = MinMaxScaler()
    nat_data = sc_minmax.fit_transform(df_nat[features].values)
    v_seq = 2
    X_nat = torch.tensor([nat_data[i:i+v_seq] for i in range(len(nat_data)-v_seq)], dtype=torch.float32)
    y_nat = torch.tensor([nat_data[i+v_seq] for i in range(len(nat_data)-v_seq)], dtype=torch.float32)
    
    # 2-Layer Bi-LSTM with high capacity
    lstm = NationalCrimeLSTM(len(features), 128, 2, len(features))
    opt = optim.Adam(lstm.parameters(), lr=0.001)
    for _ in range(800):
        out = lstm(X_nat); loss = nn.MSELoss()(out, y_nat)
        opt.zero_grad(); loss.backward(); opt.step()
    
    with torch.no_grad():
        # Recursive Audit: Predict 2022, then predict 2023
        h22_vec = lstm(torch.tensor(nat_data[:v_seq].reshape(1,v_seq,-1), dtype=torch.float32)).numpy()
        h23_seq = np.append(nat_data[1:v_seq], h22_vec, axis=0)
        p23_pure = sc_minmax.inverse_transform(lstm(torch.tensor(h23_seq.reshape(1,v_seq,-1), dtype=torch.float32)).numpy())[0]
        p22_pure = sc_minmax.inverse_transform(h22_vec)[0]
    
    feat_to_idx = {f: i for i, f in enumerate(features)}
    years = df_nat['Year'].values
    
    log_msg("\n[PHASE 1: RAW NEURAL ACCURACY (NO SCALING)]")
    for d in DOMAINS:
        idxs = [feat_to_idx[f] for f in domain_feats[d]]
        act_vals = df_nat[domain_feats[d]].sum(axis=1).values
        pred_raw_23 = sum([p23_pure[i] for i in idxs])
        pred_raw_22 = sum([p22_pure[i] for i in idxs])
        
        acc = max(0, 100 - (abs(pred_raw_23 - act_vals[4]) / act_vals[4] * 100))
        
        plt.figure(figsize=(10, 5))
        plt.plot(years, act_vals, 'bo-', label='Actual NCRB Data', linewidth=3)
        plt.plot([2021, 2022, 2023], [act_vals[2], pred_raw_22, pred_raw_23], 'rx--', label=f'Neural Audit (Acc: {acc:.1f}%)', linewidth=2)
        plt.title(f"2023 Pure Neural Audit: {d} Domain", fontsize=14)
        plt.xlabel("Year"); plt.ylabel("Total Number of Crimes"); plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR, f"audit_comparison_{d.lower().replace(' ', '_')}.png"), dpi=150)
        plt.close()

        log_msg(f"   - {d}: {int(act_vals[4]):,} (Act) vs {int(pred_raw_23):,} (Neural) | {acc:.2f}%")

    # National Total Audit
    total_act_nat = df_nat[features].sum(axis=1).values
    pred_nat_23 = p23_pure.sum()
    pred_nat_22 = p22_pure.sum()
    nat_acc = max(0, 100 - (abs(pred_nat_23 - total_act_nat[4]) / total_act_nat[4] * 100))

    plt.figure(figsize=(10, 5))
    plt.plot(years, total_act_nat, 'bo-', label='Actual India Total', linewidth=4)
    plt.plot([2021, 2022, 2023], [total_act_nat[2], pred_nat_22, pred_nat_23], 'rx--', label=f'Macro-Neural Audit (Acc: {nat_acc:.1f}%)', linewidth=3)
    plt.title(f"India-National Pure Neural Audit (Accuracy: {nat_acc:.2f}%)", fontsize=16)
    plt.xlabel("Year"); plt.ylabel("Total Number of Crimes"); plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "audit_comparison_combined.png"), dpi=150)
    plt.close()

    log_msg(f"\n[PHASE 2: MACRO-SYSTEMIC NEURAL PERFORMANCE]")
    log_msg(f"   - National Total Accuracy: {nat_acc:.2f}%")

    log_msg("\n=== PURE SCIENTIFIC AUDIT COMPLETE ===")

if __name__ == "__main__":
    run_full_spectrum_audit()
