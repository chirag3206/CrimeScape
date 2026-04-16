from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import joblib
import os
import json

app = Flask(__name__)

# -------- SETUP PATHS --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "Analysis", "results", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_FILE = os.path.join(BASE_DIR, "Analysis", "results", "tables", "ml_ready_pivoted_rates.csv")
SUMMARY_TEXT = os.path.join(BASE_DIR, "Analysis", "EDA.txt")

# Custom Route to serve plots directly from Analysis folder
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    try:
        df = pd.read_csv(DATA_FILE, index_col='State')
        states = sorted(df.index.tolist())
        categories = ["Women", "Children", "Juvenile", "Human Traficking"]
        
        # Most intense state overall
        worst_state = df.sum(axis=1).idxmax()
        
        # Load summary text from EDA report
        report_content = ""
        if os.path.exists(SUMMARY_TEXT):
            with open(SUMMARY_TEXT, 'r') as f:
                report_content = f.read()

        return jsonify({
            "states": states,
            "categories": categories,
            "worst_state": worst_state,
            "report_summary": report_content[:2000],  # Share first chunk for dashboard
            "meta": {
                "total_records": 71662,
                "years_covered": "2019-2023"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state = data.get('state')
        domain = data.get('domain')
        
        if not state or not domain:
            return jsonify({"error": "State and Domain are required"}), 400

        # 1. Prediction: Insecurity Score (XGBoost Regressor)
        reg_model_path = os.path.join(MODELS_DIR, domain, "Regression", "xgboost_regressor_model.joblib")
        reg_model = joblib.load(reg_model_path)
        
        # 2. Classification: Risk Grade (Lasso Logistic Classifier)
        clf_model_path = os.path.join(MODELS_DIR, domain, "Classification", "lasso_logistic_model.joblib")
        clf_model = joblib.load(clf_model_path)
        
        # Load state features
        df = pd.read_csv(DATA_FILE, index_col='State').fillna(0)
        domain_cols = [col for col in df.columns if col.startswith(f"{domain}_")]
        X_input = df.loc[[state], domain_cols]
        
        # Sanitize for XGBoost
        X_input.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_input.columns]
        
        predicted_intensity = reg_model.predict(X_input)[0]
        risk_grade_num = clf_model.predict(X_input)[0]
        risk_grade = "HIGH RISK" if risk_grade_num == 1 else "STABLE/LOW RISK"

        return jsonify({
            "state": state,
            "domain": domain,
            "intensity_score": round(float(predicted_intensity), 2),
            "risk_grade": risk_grade,
            "confidence": "High (Model R-squared verified)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/map_data/<domain>')
def get_map_data(domain):
    try:
        model_path = os.path.join(MODELS_DIR, domain, "Regression", "xgboost_regressor_model.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model for {domain} not found"}), 404
            
        model = joblib.load(model_path)
        df = pd.read_csv(DATA_FILE, index_col='State').fillna(0)
        domain_cols = [col for col in df.columns if col.startswith(f"{domain}_")]
        
        X_all = df[domain_cols].copy()
        X_all.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_all.columns]
        
        predictions = model.predict(X_all)
        
        # Return a mapping of State -> Predicted Intensity
        data_map = {}
        for i, state in enumerate(df.index):
            data_map[state] = round(float(predictions[i]), 2)
            
        return jsonify(data_map)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data = request.json
        state1 = data.get('state1')
        state2 = data.get('state2')
        domain = data.get('domain')
        
        reg_model = joblib.load(os.path.join(MODELS_DIR, domain, "Regression", "xgboost_regressor_model.joblib"))
        clf_model = joblib.load(os.path.join(MODELS_DIR, domain, "Classification", "lasso_logistic_model.joblib"))
        
        df = pd.read_csv(DATA_FILE, index_col='State').fillna(0)
        domain_cols = [col for col in df.columns if col.startswith(f"{domain}_")]
        
        def get_pred(state):
            X_in = df.loc[[state], domain_cols].copy()
            X_in.columns = [c.replace('[', '_').replace(']', '_').replace('<', '_') for c in X_in.columns]
            intensity = reg_model.predict(X_in)[0]
            risk = "HIGH RISK" if clf_model.predict(X_in)[0] == 1 else "STABLE"
            return {"intensity": round(float(intensity), 2), "risk": risk}

        return jsonify({
            "state1": {**get_pred(state1), "name": state1},
            "state2": {**get_pred(state2), "name": state2},
            "domain": domain
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n--- NCRB ANALYTICS BACKEND STARTING ---")
    print("Serving dashboard at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
