# 🛡️ Crime-Scape: Vision 2030 Intelligence Suite

An enterprise-grade, interactive web application for analyzing, auditing, and forecasting crime trends in India. Powered by Deep Learning, Crime-Scape bridges the reporting gap using NCRB data (2019-2023) to project high-priority domain intensities through 2030.

![Dashboard Preview](https://via.placeholder.com/1000x500.png?text=Crime-Scape+Vision+2030+Deep+Learning+Suite)

## 🚀 Key Intelligence Features

- **📉 Vision 2030 Long-Range Forecasting**: A recursive **LSTM (Long Short-Term Memory)** engine that analyzes 500+ historical vectors to project standardized crime trends from 2024 to 2030.
- **🛡️ Neural Risk Audit (ANN)**: A multi-layer deep risk classifier using **Magnitude + Velocity (Momentum)** logic. It flags states based not just on current volume, but on the speed of their crime trajectory.
- **🌉 GNN Spatial Spillover Mapping**: utilizes **Graph Neural Networks** to measure cross-border influence and regional trend propagation across 36 States and UTs.
- **⚖️ 4-Domain specialized Protection**: Deep-dive analysis across four critical protection spheres:
  - **Women's Safety**
  - **Child Protection**
  - **Juvenile Justice Interaction**
  - **Human Trafficking Surveillance**
- **🗺️ Geospatial Intensity Audit**: High-fidelity interactive heatmapping for regional risk assessment.

## 🧠 The Neural Architecture (4 Pillars)

| Pillar | Model Type | Purpose |
| :--- | :--- | :--- |
| **I. Temporal Core** | Stacked LSTM | Recursive forecasting from 2024-2030. |
| **II. Audit Engine** | MLP (ANN) | Dynamic Risk Tiering (Low, Stable, High, Critical). |
| **III. Spatial Index** | GCN (GNN) | Measuring the 'Spillover' effect across shared borders. |
| **IV. Profiling** | Autoencoders | Extracting 'State Fingerprints' for sociological clustering. |

## ⚙️ Mathematical Normalization
To maintain data integrity across thousands of rows, the system employs a synchronized **Intensity Indexing Factor (1/55.0)**. This allows raw case counts (Thousands) and per-capita rates (Per 100k) to be analyzed on a singular, realistic magnitude scale (~230-350 for high-risk regions).

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch (LSTM, GCN, ANN)
- **Machine Learning**: Scikit-learn, XGBoost, Ridge/Lasso Ensemble
- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript, Chart.js
- **Data Engineering**: Pandas, NumPy

## 📁 Project Structure

```text
├── Analysis/
│   ├── scripts/
│   │   ├── DeepLearning/  # Custom Neural Engines (LSTM, ANN, GNN, AE)
│   │   └── Models_ML/      # Classical ML pipelines
│   └── results/           # Synchronized Intelligence (Forecasts, Spatial, Risk)
├── Models/                # Serialized Weights (.pth, .joblib)
├── static/                # Premium Glassmorphism UI (CSS/JS)
├── templates/             # Dashboard Templates
└── app.py                 # Neural Core API & Intelligence Gateway
```

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chirag3206/CrimeScape.git
   ```
2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Intelligence Gateway**:
   ```bash
   python app.py
   ```

## 📈 Data Source
Intelligence derived and synthesized from the official [National Crime Records Bureau (NCRB)](https://ncrb.gov.in/) datasets.

---
Created with ❤️ by Antigravity AI for Public Safety Analytics.
