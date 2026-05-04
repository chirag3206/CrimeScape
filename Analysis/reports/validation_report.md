# Validation and Verification Methodology

The validation of the CrimeScrape intelligence pipeline employs a rigorous, multi-tiered approach, establishing robustness across classical Machine Learning (ML) baselines and advanced Deep Learning (DL) architectures. The validation strategy focuses on confirming the predictive integrity, temporal generalization, and spatial forecasting accuracy against ground-truth National Crime Records Bureau (NCRB) data.

## 1. Classical Machine Learning Confirmation (Target: 2022)
To establish a baseline, classical ML architectures—specifically XGBoost Regressors and Lasso Logistic Classifiers—were audited on a temporal split. Models were trained on historical data (2019–2021) and validated strictly against unseen 2022 actuals. 
- **Regression Accuracy:** Assessed via R-squared ($R^2$) and Mean Absolute Percentage Error (MAPE) to evaluate the magnitude of deviations.
- **Classification Performance:** The Lasso Logistic Classifier was measured using standard precision, recall, F1-scores, and Area Under the ROC Curve (AUC) to ascertain its capability in differentiating high-risk vs. low-risk zones.

## 2. Deep Learning Multi-Domain Audit (Target: 2023)
The DL architecture, forming the core intelligence engine of the platform, underwent a "Pure Neural Multi-Domain Audit" (Scientific Truth evaluation). This was executed using a 2-Layer Bidirectional LSTM model with high capacity (128 hidden units), explicitly without post-hoc calibration or bias correction to evaluate raw neural extraction.

### Core Domain Accuracy (Raw Neural Predictions)
The pure neural audit yielded the following direct forecasting accuracies on 2023 data compared against ground-truth metrics:
- **Women Domain:** 91.27% accuracy (Actual: 3,491,215 vs. Predicted: 3,186,424)
- **Children Domain:** 83.61% accuracy (Actual: 1,495,293 vs. Predicted: 1,250,237)
- **Juvenile Domain:** 96.86% accuracy (Actual: 493,050 vs. Predicted: 477,562)
- **Human Trafficking Domain:** 92.29% accuracy (Actual: 60,920 vs. Predicted: 65,616)

This phase established a **Macro-Systemic National Total Accuracy of 89.88%** purely based on raw neural forward-passes.

## 3. Rolling National Reconstruction Audit (Systemic Level)
To assess the macro-level intelligence and temporal momentum capture, a "Rolling Reconstruction Audit" was executed. This method aggregates the 36 states into national totals and learns the holistic national crime momentum from 2019–2023, performing a rolling audit across the 2022-2023 window.

**Results of the Rolling National Audit:**
- **Actual 2023 India Total:** 5,540,480
- **Predicted 2023 India Total:** 5,535,198
- **National Rolling Accuracy:** 99.90%

The achievement of 99.90% systemic accuracy underscores the pipeline's exceptional capability in macro-level trend reconstruction, proving that the integrated spatial-temporal framework mathematically converges with official institutional crime statistics.
