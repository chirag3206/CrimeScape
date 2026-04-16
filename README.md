# 🛡️ Crime-Scape: NCRB Analytics Dashboard

An advanced, interactive web application for analyzing and forecasting crime trends in India using National Crime Records Bureau (NCRB) data (2019-2023).

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Crime-Scape+Analytics+Dashboard)

## 🚀 Key Features

- **📊 Interactive Visualizations**: Dynamic Choropleth maps and trend charts using Plotly.
- **🤖 ML Risk Forecasting**: Predictive modeling (XG Boost & Lasso Logistic Regression) to assess state-level insecurity scores and risk grades.
- **⚖️ Comparative Intelligence**: Side-by-side analysis of different states and crime domains (Women, Children, Juvenile, Human Trafficking).
- **📑 Automated Insights**: Generates structured summaries from extensive data analysis.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Data Science**: Pandas, Scikit-learn, XGBoost
- **Visualization**: Plotly, Leaflet.js
- **Models**: Pre-trained Joblib models for real-time inference

## 📁 Project Structure

```text
├── Analysis/          # Data processing notebooks & result tables
├── Models/            # Trained ML models (Regression/Classification)
├── Scrapping/         # PDF extraction scripts for NCRB data
├── static/            # CSS, JS, and Images
├── templates/         # HTML templates
├── app.py             # Main Flask application
└── requirements.txt    # Project dependencies
```

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chirag3206/CrimeScape.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
4. **Access the dashboard**: Open `http://127.0.0.1:5000` in your browser.

## 📈 Data Source
Data processed from the official [National Crime Records Bureau (NCRB)](https://ncrb.gov.in/) annual reports.

---
Created with ❤️ for Public Safety Analytics.
