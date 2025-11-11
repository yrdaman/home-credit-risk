# Home Credit Default Risk — Risk Scoring System

End-to-end machine learning system predicting default probability of loan applicants using the Home Credit multi-table dataset.

### Goal
Estimate the probability of default at application time, then map that probability to risk bands + recommended actions.

# Data Folder

Dataset: Home Credit Default Risk (Kaggle)
https://www.kaggle.com/competitions/home-credit-default-risk/data

NOTE:
Currently awaiting Kaggle account verification to download.
Once verified, place all .csv files here (not committed to Git).

### ML Approach
- Multi-table → aggregated feature engineering
- Baselines: Logistic Regression → LightGBM
- Class imbalance handling + Stratified K-Fold CV
- Probability calibration (Isotonic / Platt)
- Explainability with SHAP

### Deployment
FastAPI service with `/score` endpoint returning:
- probability of default (0-1)
- risk band (Green / Amber / Red)
- top feature explanations

### Target Metric
ROC-AUC ≥ 0.80

### Future Work
- Profit optimization threshold
- drift detection
- fairness segmentation

### Deployment Target
Railway.app
