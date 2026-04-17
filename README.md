# 🛡️ Crime-Scape: Vision 2030 Intelligence Suite

An enterprise-grade, interactive web application for analyzing, auditing, and forecasting crime trends in India. Powered by Deep Learning, Crime-Scape bridges the reporting gap using NCRB data (2019-2023) to project high-priority domain intensities through 2030.

![Dashboard Preview](https://via.placeholder.com/1000x500.png?text=Crime-Scape+Vision+2030+Deep+Learning+Suite)

## 🚀 Key Intelligence Features

- **📉 Vision 2030 Multi-Stream Forecasting**: A recursive **LSTM (Long Short-Term Memory)** engine projecting trajectories from 2024 to 2030. Users can pivot between a **Combined Intensity Index** and specific high-fidelity trajectories for Women, Children, Juvenile, and Trafficking domains.
- **📊 National Benchmark Overlay**: A bi-linear visualization suite that overlays a **National Average Baseline** (Dashed) atop the Regional Trend (Solid). This allows for instant comparative analysis of state performance against the country's average.
- **🛡️ Neural Risk Audit (ANN)**: A multi-layer deep risk classifier using **Magnitude + Velocity (Momentum)** logic. It flags states based not just on current volume, but on the speed and acceleration of their crime trajectory through 2030.
- **🌉 GNN Spatial Spillover Mapping**: Utilizes **Graph Neural Networks** to measure cross-border influence and regional trend propagation across 36 States and UTs.
- **🗺️ Geospatial Intensity Audit**: High-fidelity interactive heatmapping and comparison lab for real-time regional assessment.

## 🧠 The Neural Architecture (4 Pillars)

| Pillar | Model Type | Purpose |
| :--- | :--- | :--- |
| **I. Temporal Core** | Stacked LSTM | Recursive forecasting with multi-domain intelligence streams. |
| **II. Audit Engine** | MLP (ANN) | Dynamic Risk Tiering based on Magnitude & Trend Momentum. |
| **III. Spatial Index** | GCN (GNN) | Measuring the 'Spillover' effect across shared state borders. |
| **IV. Profiling** | Autoencoders | Extracting 'State Fingerprints' for sociological clustering. |

## ⚙️ Mathematical Normalization: The "Delhi Factor"
To maintain data integrity and cross-domain parity, the system employs a synchronized **Intensity Indexing Factor (1/55.0)**. 
- **Standardized Rates**: Women and Child crime are analyzed via established per-100k rates.
- **Normalized Counts**: Juvenile and Trafficking raw counts are divided by 55.0 to approximate an "Impact-per-100k" magnitude.
This ensures that all trajectories—whether regional or national average baseline—are plotted on a singular, realistic comparison scale.

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch (LSTM, GCN, ANN) - Bi-linear comparative data streams.
- **Backend**: Flask (Python) - High-performance dual-channel JSON API.
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript, Chart.js (Dual-dataset rendering).
- **Data Engineering**: Pandas, NumPy (Multi-region temporal aggregation).

## 📁 Project Structure

```text
├── Analysis/
│   ├── scripts/
│   │   ├── DeepLearning/  # Custom Neural Engines (LSTM, ANN, GNN, AE)
│   │   └── Models_ML/      # Classical ML pipelines for ensemble validation
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
Intelligence derived and synthesized from the official [National Crime Records Bureau (NCRB)](https://ncrb.gov.in/) datasets through recursive deep learning projection.

---
Created by **Antigravity AI** for Advanced Public Safety Intelligence.🛡️🇮🇳🛰️
