import os

BASE_DIR = r"D:\College\Projects\PRJ-3\Crimes"
OUTPUT_FILE = os.path.join(BASE_DIR, "Comprehensive_CrimeScrape_Report.md")

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def generate_viva_qa():
    return """## Part 2: Comprehensive Viva Preparation Q&A

This section contains highly detailed Questions and Answers covering all technical aspects of the project, designed specifically for academic Viva Voce preparation.

### 1. Deep Learning Architecture (Vision 2030)

**Q: Why was Deep Learning (LSTM) chosen over traditional Time Series models like ARIMA for Vision 2030 forecasting?**
A: Traditional models like ARIMA are univariate and struggle with high-dimensional, multivariate datasets where features are highly correlated. Crime data has complex spatial and temporal dependencies. A Stacked LSTM (Long Short-Term Memory) network can retain historical momentum and capture non-linear relationships across hundreds of crime features simultaneously. Our implementation uses a recursive rolling window approach, predicting 2024 from 2021-2023, then using the 2024 prediction to predict 2025, enabling long-range projections.

**Q: Explain the architecture and purpose of the Spatial GNN.**
A: The Spatial Graph Neural Network (GNN), specifically a Graph Convolutional Network (GCN), is used to model the 'Spillover Effect' of crime across state borders. 
- **Nodes:** The 36 States and UTs.
- **Edges:** Shared geographical borders defined by an Adjacency Matrix.
- **Features:** Projected 2030 crime intensities.
The GCN performs neighborhood feature aggregation (H = D^(-1/2) * A * D^(-1/2) * X * W). It allows us to mathematically quantify how crime in a high-intensity state (like UP or Maharashtra) influences its neighboring states, creating a Regional Spillover Index.

**Q: How does the ANN Risk Auditor work and what is the 'Hybrid Score'?**
A: The ANN Risk Auditor is a Multi-Layer Perceptron (128->64->32->1) that acts as a binary classifier (High Risk vs Low Risk). It was trained on historical data using the 75th percentile of total crime as the high-risk threshold.
For the 2030 projections, we don't just rely on the absolute predicted volume (Magnitude). We calculate the Trend Velocity (the rate of change from 2024 to 2030). The Hybrid Risk Score is a weighted combination: 60% Neural Magnitude + 80% Trend Momentum. This ensures that a state with low overall crime but rapidly escalating growth (high momentum) is correctly flagged as Critical.

**Q: What is the purpose of the Autoencoder in this project?**
A: The Autoencoder is used for sociological profiling and dimensionality reduction. We have ~500 crime features per state. The Autoencoder compresses these 500 features into a 16-dimensional "Latent Space" or "State Fingerprint" using a bottleneck architecture (Encoder: 500->128->64->16). Two states with similar 16D fingerprints have similar sociological crime patterns, regardless of their total population or overall crime volume. This enables deep clustering and comparative analysis.

### 2. Machine Learning Pipeline

**Q: Which Machine Learning algorithms were used for the historical analysis (2019-2023) and why?**
A: We implemented an ensemble suite:
1. **XGBoost Regressor:** Used for predicting exact crime intensity scores based on state features. XGBoost handles non-linearities and tabular data exceptionally well.
2. **Lasso Logistic Regression:** Used for binary risk classification. The L1 penalty (Lasso) inherently performs feature selection, driving the weights of irrelevant crime factors to zero, which helps us identify the core indicators of high risk.
3. **DBSCAN:** Used for unsupervised clustering of states. DBSCAN groups states based on feature density and can identify statistical outliers (noise) seamlessly without needing a predefined number of clusters (like K-Means).

**Q: How did you evaluate the performance of the models?**
A: We used a strict temporal split. The ML models were trained on 2019-2021 data and validated against 2022 actuals. 
- Regression was evaluated using R-Squared (R2) and Mean Absolute Percentage Error (MAPE).
- Classification was evaluated using Accuracy, F1-Score, and ROC-AUC.
- The DL models underwent a "Pure Neural Audit" where they were tested against 2023 ground-truth data, achieving a Macro-Systemic National Accuracy of 99.90% for the national aggregate and >85% for individual domains.

### 3. Data Engineering & Processing

**Q: How was the raw data extracted from the NCRB PDFs?**
A: We used the `pdfplumber` library in Python to extract text from specific pages of the NCRB PDF reports. Regular expressions (Regex) were heavily utilized to parse the unstructured text, handling variations in state names, missing columns, and merging multiline entries. We created 10 separate scraping scripts to accurately extract data for different IPC/SLL sections across the 5 domains (Women, Children, etc.).

**Q: What is "The Delhi Factor" or Normalization Strategy mentioned in the documentation?**
A: Direct comparison of raw crime counts between a massive state like UP and a small UT like Delhi is statistically invalid. We normalized the data into "Intensity Rates". 
- For Women & Children crimes, we used standard Per-100k population rates using 2021 census projections. 
- For Juvenile & Trafficking, where raw counts were used, we applied a synchronization constant (divided by 55.0) to approximate an "Impact-per-100k" magnitude. This ensures all domains can be visualized on the same dashboard scale without distortion.

**Q: How did you handle Explainable AI (XAI)?**
A: We integrated the SHAP (SHapley Additive exPlanations) library. SHAP uses game theory to calculate the exact contribution of each feature to a specific prediction. We built a custom `CrimeExplainer` utility that not only calculates these SHAP values but translates them into a professional text narrative (Executive Intelligence Briefing) explaining *why* a state received a specific risk score and providing automated policy directives based on the top contributing crimes.

### 4. Web Application & Deployment

**Q: Explain the technology stack of the Dashboard.**
A: The backend is powered by Flask (Python), exposing 7 RESTful API endpoints. The ML/DL models are serialized using `joblib` and PyTorch's native tensor operations. 
The frontend uses Vanilla HTML5, CSS3, and JavaScript without heavy framework bloat. It features a modern "Glassmorphism" design system. Visualizations are rendered using Chart.js, and the geospatial intensity maps are powered by Leaflet.js with CartoDB dark tiles and India GeoJSON polygons.

**Q: How does the application handle anomaly detection in real-time?**
A: We integrated an `IsolationForest` model that runs an audit on the dataset. It calculates anomaly scores for each state based on their multidimensional feature vectors. If a state deviates significantly from the national distribution (a statistical outlier, like Delhi's extreme density), it is flagged and displayed dynamically on the dashboard header.
"""

def generate_report():
    print("Generating comprehensive report...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write("# 🛡️ CrimeScrape: Vision 2030 Intelligence Suite\n")
        out.write("## The Ultimate Project Report, Documentation, and Viva Guide\n\n")
        
        out.write("> **Notice:** This is a comprehensive, autogenerated 4000+ line document encompassing the entire project architecture, source code, data pipelines, Machine Learning metrics, Deep Learning mathematics, and Viva Voce preparation materials.\n\n")
        
        # Part 1: Project Overview (From README and Analysis)
        out.write("## Part 1: Executive Summary & Project Architecture\n\n")
        readme_content = read_file(os.path.join(BASE_DIR, "README.md"))
        out.write(readme_content)
        out.write("\n\n")
        
        out.write("### Data Pipeline Architecture\n")
        out.write("1. **Ingestion**: pdfplumber parses official NCRB PDFs (2019-2023).\n")
        out.write("2. **ETL**: Data is cleaned, state names are normalized, and a master long-format CSV is generated (`merge_all_data.py`).\n")
        out.write("3. **Pivoting**: Long data is pivoted into a wide matrix (State x Features) for ML models (`advanced_eda.py`).\n")
        out.write("4. **Temporal Stacking**: Data is pivoted into (State x Year x Features) sequences for LSTM forecasting (`lstm_data_prep.py`).\n\n")
        
        # Part 2: Viva Q&A
        out.write(generate_viva_qa())
        out.write("\n\n")
        
        # Part 3: Analytical Reports & Validation
        out.write("## Part 3: Deep Learning Research & Validation Reports\n\n")
        
        reports_to_include = [
            ("Vision 2030 Deep Learning Report", r"Analysis\reports\Vision2030_DL_Report.txt"),
            ("Deep Learning Actual Accuracies", r"Analysis\reports\dl_actual_accuracies.md"),
            ("Validation & Verification Methodology", r"Analysis\reports\validation_report.md"),
            ("Advanced EDA Summary", r"Analysis\EDA.txt") # Assuming EDA.txt exists as EDA_Advanced_Summary.txt
        ]
        
        for title, rel_path in reports_to_include:
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path) or os.path.exists(full_path.replace("EDA.txt", "EDA_Advanced_Summary.txt")):
                actual_path = full_path if os.path.exists(full_path) else full_path.replace("EDA.txt", "EDA_Advanced_Summary.txt")
                out.write(f"### {title}\n")
                out.write("```text\n")
                out.write(read_file(actual_path))
                out.write("\n```\n\n")
        
        # Part 4: Complete Codebase Walkthrough
        out.write("## Part 4: Complete Codebase Walkthrough & Source Code Integration\n\n")
        out.write("This section details every critical script in the project, explaining its exact purpose, followed by the raw source code for complete transparency and study.\n\n")
        
        # Define the directories and files to walk through
        critical_files = [
            ("Core Web Application", "app.py", "The main Flask backend server handling API requests, routing, and model inference."),
            ("AI & Explainability Utilities", "ai_utils.py", "Contains the SHAP explainer engine for XAI and the Isolation Forest anomaly detector."),
            ("LSTM Temporal Forecaster", r"Analysis\scripts\DeepLearning\LSTM\lstm_vision_2030.py", "The Pillar I architecture. 2-layer LSTM predicting 2024-2030 trajectories."),
            ("ANN Risk Auditor", r"Analysis\scripts\DeepLearning\ANN\ann_risk_grading.py", "The Pillar II architecture. Multi-layer perceptron generating hybrid risk scores."),
            ("Spatial GNN Intelligence", r"Analysis\scripts\DeepLearning\SpatialGNN\gnn_spatial_intelligence.py", "The Pillar III architecture. GCN mapping regional spillover across state borders."),
            ("Autoencoder Profiler", r"Analysis\scripts\DeepLearning\Encoders\autoencoder_profiling.py", "The Pillar IV architecture. Compresses 500 features into 16D state fingerprints."),
            ("ML XGBoost & Random Forest", r"Analysis\scripts\Models_ML\random_forest_xgboost_regressor.py", "Regression models predicting crime intensities and extracting feature importance."),
            ("ML Lasso Logistic Classifier", r"Analysis\scripts\Models_ML\lasso_logistic_classifier.py", "High-risk binary classification with L1 regularization for feature selection."),
            ("ML DBSCAN Clustering", r"Analysis\scripts\Models_ML\dbscan_clustering.py", "Unsupervised clustering of states to find sociological archetypes."),
            ("Master Data Merger", r"Analysis\scripts\Preprocessing & EDA\merge_all_data.py", "Aggregates all scraped CSVs into a single master dataset, normalizing state names."),
            ("Advanced EDA Engine", r"Analysis\scripts\Preprocessing & EDA\advanced_eda.py", "Calculates crime rates, checks volatility, creates correlation matrices, and pivots data for ML."),
            ("Pure Neural DL Audit", r"Analysis\scripts\run_dl_confirmation.py", "Validates the LSTM against actual 2023 NCRB data to establish the 99.9% accuracy baseline."),
            ("Classical ML Audit", r"Analysis\scripts\run_all_validation.py", "Validates XGBoost and Logistic models on a strict temporal split (Train: 19-21, Test: 22-23)."),
            ("Production Artifact Generator", r"Analysis\scripts\generate_production_artifacts.py", "Synchronizes the final Neural Risk and Spatial maps for dashboard consumption."),
            ("Frontend HTML Dashboard", r"templates\index.html", "The user interface structure containing the 5 intelligent tabs."),
            ("Frontend JavaScript Logic", r"static\js\script.js", "Handles dynamic API fetching, Chart.js rendering, and Leaflet map interactions."),
            ("Frontend Premium CSS", r"static\css\style.css", "Glassmorphism styling, responsive grid layouts, and custom animations.")
        ]
        
        # Add scraping scripts dynamically
        scraping_dir = os.path.join(BASE_DIR, "Scrapping")
        if os.path.exists(scraping_dir):
            for file in os.listdir(scraping_dir):
                if file.endswith(".py"):
                    critical_files.append((
                        f"Scraping Engine: {file}", 
                        os.path.join("Scrapping", file), 
                        f"Extracts raw tabular data from NCRB PDFs using pdfplumber and regex for {file.split('_')[0]}."
                    ))

        for title, rel_path, desc in critical_files:
            full_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(full_path):
                out.write(f"### {title}\n")
                out.write(f"**Filepath:** `{rel_path}`\n\n")
                out.write(f"**Description:** {desc}\n\n")
                
                lang = "python"
                if rel_path.endswith(".html"): lang = "html"
                elif rel_path.endswith(".js"): lang = "javascript"
                elif rel_path.endswith(".css"): lang = "css"
                
                out.write(f"```{lang}\n")
                out.write(read_file(full_path))
                out.write("\n```\n\n")
                out.write("---\n\n")
                
    print(f"Report successfully generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
