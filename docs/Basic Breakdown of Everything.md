## Data
- Read sensor values over time (temp, flow, vibration). 
- Keep a short history of this info

## Lightweight Models (simple)
- **Statistical Thresholds (z-score, EWMA):** Learn what normal looks like per sensor and alert when a value or moving average strays too far. Cheap and fast
- **Isolation Forest:** Randomly splits data so true anomalies get isolated quickly. As a result, these anomalies score high which is a good default for tabular sensor features.
- **k-Nearest Neighbors(KNN):** Anomaly score is based off if a point is far from the closest past normal, these are flagged. KNN uses simple distance check on features.
- **One-Class SVM:** Draws a tight boundary around normal data; points outsides are anomalies but they need careful scaling. 
- **Compact Autoencoder:** Learns to reconstruct normal patterns. Large reconstruction error causes a anomaly. The reconstructions need to be kept small to stay CPU-friendly.
- **Logistic Regression (supervised baseline):** If you have labels, a linear classifier on engineered features. This model is fast with with interpretable coefficients. 
- **Shallow Gradient Boosting/ XG Boost (supervised baseline):** Small tree ensemble for labeled fault vs normal. Strong tabular baseline but with limited depth. 

## Alerting
 - Each model outputs a score that says how abnormal the latest data looks, the high the more unusual. 
 - Use a cutoff to detect anomalies. 
 - If score > cutoff, flag anomaly 
