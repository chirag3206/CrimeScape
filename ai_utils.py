import shap
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

class CrimeExplainer:
    def __init__(self):
        self.explainers = {}

    def get_explanation(self, model, X_input, feature_names):
        """
        Generates a SHAP explanation for a single prediction.
        """
        try:
            # Check if we already have an explainer for this model instance
            model_id = id(model)
            if model_id not in self.explainers:
                # XGBoost models work best with TreeExplainer
                self.explainers[model_id] = shap.TreeExplainer(model)
            
            explainer = self.explainers[model_id]
            shap_values = explainer.shap_values(X_input)
            
            # For regression, shap_values is often a simple array
            # If it's a list (for some multi-output), take the first one
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Map features to values
            contributions = []
            for i, name in enumerate(feature_names):
                val = float(shap_values[0][i])
                contributions.append({
                    "feature": name.replace('_', ' '),
                    "value": round(val, 4)
                })
            
            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x['value']), reverse=True)
            
            # Return top 10 influencers
            return contributions[:10]
        except Exception as e:
            print(f"❌ SHAP Error: {str(e)}")
            return []

    def generate_narrative(self, contributions, state, domain):
        """
        Converts SHAP contributions into a professional textual narrative.
        """
        if not contributions:
            return "Insufficient data to generate a detailed intelligence narrative."

        positives = [c for c in contributions if c['value'] > 0]
        negatives = [c for c in contributions if c['value'] < 0]

        # Constructing the Narrative
        narrative = f"### Executive Intelligence Briefing: {state}\n\n"
        
        if len(positives) > 0:
            top_p = positives[0]['feature']
            narrative += f"The neural engine has identified **{top_p}** as the primary driver for the projected intensity in this region. "
            if len(positives) > 1:
                narrative += f"Additional pressure is observed from **{positives[1]['feature']}**. "
        
        narrative += "\n\n#### Key Findings:\n"
        for c in positives[:3]:
            impact = "Significant" if abs(c['value']) > 0.1 else "Moderate"
            narrative += f"- **{c['feature']}**: Shows a {impact.lower()} upward trajectory, contributing nearly {abs(round(c['value']*100, 1))}% to the risk weight.\n"

        if negatives:
            narrative += "\n#### Stability Buffers:\n"
            for c in negatives[:2]:
                narrative += f"- Observed stability in **{c['feature']}** is currently acting as a statistical buffer, preventing a further score escalation.\n"

        narrative += f"\n**Policy Recommendation**: Priority should be given to mitigating the growth of {positives[0]['feature']} to stabilize the overall {domain} security index by 2026."

        # Add Strategic Action Plan
        narrative += "\n\n#### 🛡️ Strategic Action Plan (Directives)\n"
        for c in positives[:3]:
            # Finding a match in the mapping
            advice = "Initiate comprehensive data-driven policing and increase the officer-to-citizen ratio to national benchmarks."
            for key, val in PROPOSED_INTERVENTIONS.items():
                if key.lower() in c['feature'].lower():
                    advice = val
                    break
            
            narrative += f"- **Target: {c['feature']}** → {advice}\n"

        return narrative

# Strategic Action Mapping
PROPOSED_INTERVENTIONS = {
    "Cruelty": "Implement community-based domestic violence monitoring and rapid-response support units.",
    "Assault": "Increase regional patrol density and improve high-visibility lighting in identified incident clusters.",
    "Abduction": "Deploy advanced signal intelligence for rapid tracking and strengthen border/transit point surveillance.",
    "Rape": "Establish Fast-Track Special Courts (FTSCs) and enhance forensic evidence collection infrastructure.",
    "Stalking": "Upgrade digital surveillance frameworks and implement specialized gender-sensitivity training for local law enforcement.",
    "Trafficking": "Strengthen cross-border intelligence sharing and establish regional anti-trafficking units.",
    "Theft": "Incentivize community-based smart surveillance (CCTV) and modernize beat-patrol coordination.",
    "Murder": "Expand forensic laboratory capacity and implement conflict-resolution programs in high-tension districts.",
    "Riots": "Strengthen local intelligence networks and deploy rapid-deployment forces to identified hotspots.",
    "Default": "Initiate comprehensive data-driven policing and increase the officer-to-citizen ratio to national benchmarks."
}

class AnomalyDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.features = []

    def load_and_train(self):
        try:
            if not os.path.exists(self.data_path):
                return
            
            df = pd.read_csv(self.data_path, index_col='State').fillna(0)
            self.features = df.columns.tolist()
            self.model.fit(df)
            self.is_trained = True
            print("✅ Anomaly Detector trained successfully.")
        except Exception as e:
            print(f"❌ Anomaly Training Error: {str(e)}")

    def get_anomalies(self):
        if not self.is_trained:
            self.load_and_train()
        
        try:
            df = pd.read_csv(self.data_path, index_col='State').fillna(0)
            scores = self.model.decision_function(df)
            preds = self.model.predict(df) # -1 for anomaly, 1 for normal
            
            results = []
            for i, state in enumerate(df.index):
                if preds[i] == -1:
                    results.append({
                        "state": state,
                        "score": round(float(scores[i]), 4),
                        "type": "STATISTICAL OUTLIER"
                    })
            return results
        except Exception as e:
            print(f"❌ Anomaly Detection Error: {str(e)}")
            return []

# Singleton instances
explainer_engine = CrimeExplainer()
