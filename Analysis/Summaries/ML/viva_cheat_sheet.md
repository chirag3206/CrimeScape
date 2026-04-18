# Viva Cheat Sheet: Project CrimeScrape

## Top 5 Quick-Fire Questions

**Q1: Why did you use multiple models instead of just one?**
*   **A**: We used a "Diversity of Perspective" approach. **Regressors** give us the quantitative *intensity*, **Classifiers** give us a simplified *risk grade* for high-level decisions, and **Clustering** tells us about the *structural similarity* between states. No single model captures all three dimensions.

**Q2: What was the most critical step in your data preprocessing?**
*   **A**: **Normalization per 100,000 population.** Without this, the model would simply learn population size rather than crime patterns. For example, UP (high population) might have high volume, but Delhi (high density) often has higher intensity.

**Q3: Why XGBoost over simple Linear Regression?**
*   **A**: Linear regression assumes that if Crime A increases, Total Crime increases at a constant rate. **XGBoost (Ensemble)** handles non-linear relationships and interactions (e.g., the combined effect of high unemployment and low literacy) that linear models miss.

**Q4: Why DBSCAN for clustering? Why not K-Means?**
*   **A**: Two reasons:
    1.  **Shape Independence**: K-Means assumes clusters are spherical; DBSCAN can find clusters of any shape.
    2.  **Outlier Detection**: DBSCAN explicitly identifies "Noise" points (states that don't fit any pattern), which is vital for finding extreme outliers in crime data.

**Q5: How did EDA influence your ML design?**
*   **A**: Our correlation heatmaps showed that many crime sub-categories were 90%+ correlated. This led us to use **Lasso Regression**, which automatically handles multicollinearity by zeroing out redundant features.

---

## Detailed Model Explanations (Simple Terms)

### 1. Linear / Ridge / Lasso (The Baseline)
*   **Simple Explanation**: "It's like drawing a straight line through the data points."
*   **Key takeaway**: Lasso is the 'Minimalist'—it only keeps the features that actually matter.

### 2. Random Forest / XGBoost (The Team)
*   **Simple Explanation**: "It's like asking 100 experts to vote on the crime level and taking the average."
*   **Key takeaway**: This reduces errors and prevents the model from being biased by one single year or state.

### 3. Logistic Regression (The Gatekeeper)
*   **Simple Explanation**: "It doesn't predict a number; it predicts a category (0 or 1)."
*   **Key takeaway**: We used this to 'label' states so officials can immediately see which ones are in the 'High-Risk' zone.

### 4. DBSCAN (The Grouping Specialist)
*   **Simple Explanation**: "It looks for 'crowds' of states that have similar crime profiles."
*   **Key takeaway**: It helps in peer-group analysis. If Kerala and Tamil Nadu are in the same cluster, they can share similar crime-prevention strategies.

---

## Technical Metrics to Remember
*   **R-squared (Accuracy)**: Used for Regressors. Measures how much of the crime variation our model explains (Target: >0.8).
*   **Silhouette Score**: Used for Clustering. Measures how "pure" our clusters are.
*   **Coefficient of Variation (CV)**: Used in EDA to measure "Volatility"—how much crime fluctuates year-to-year.

---
*Prepared for Project CrimeScrape Viva/Presentation*
