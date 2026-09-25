# Telco Customer Churn Prediction

A churn-prediction pipeline built on the IBM Telco Customer Churn dataset (7,043 customers, 21 raw fields). The notebook trains and compares three classifiers, calibrates the best-performing one, tunes its decision threshold, and explains its predictions with SHAP.

## Files
.
| File | Description |
|---|---|
| `Churn_Prediction_v2.ipynb` | Full pipeline: EDA → cleaning → feature engineering → model training/comparison → calibration → threshold tuning → SHAP explainability |

## Dataset

[WA_Fn-UseC_-Telco-Customer-Churn.csv](https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv) — one row per customer, target column `Churn` (Yes/No), ~26.5% positive class (imbalanced).

## Pipeline overview

1. **Cleaning** — `TotalCharges` coerced to numeric (11 blank strings → NaN → filled); `customerID` dropped as a non-predictive identifier.
2. **Feature engineering**
   - `avg_monthly_spend` = `TotalCharges / tenure`, falling back to `MonthlyCharges` for brand-new (`tenure == 0`) customers
   - `tenure_group` — binned tenure (0–6, 6–12, 12–24, 24–48, 48+ months)
   - `service_count` — count of add-on services subscribed to
   - `month_to_month` — binary flag for the highest-churn-risk contract type
3. **Split** — 70/15/15 train/validation/test, stratified on `Churn`.
4. **Preprocessing** — `StandardScaler` on numeric features, `OneHotEncoder` on categoricals, fit only on the training fold inside an sklearn `Pipeline` (no leakage).
5. **Models compared** — Logistic Regression, Random Forest, XGBoost, all trained with class-imbalance handling (`class_weight="balanced"` for LR/RF, `scale_pos_weight` for XGBoost).
6. **Model selection** — evaluated on Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier score. XGBoost is carried forward for its calibration quality (lowest Brier score), since the deliverable is a probability-based risk score rather than a fixed yes/no flag.
7. **Calibration** — `CalibratedClassifierCV` (isotonic, 5-fold) applied to XGBoost.
8. **Threshold tuning** — decision threshold optimized for F1 on the validation set, re-run specifically on the *calibrated* probabilities (best threshold ≈ 0.25) rather than reusing the raw model's threshold or a default 0.50.
9. **Explainability** — SHAP `TreeExplainer` for global feature importance and a per-customer waterfall plot.
10. **Output** — per-customer churn probability mapped to a LOW / MEDIUM / HIGH risk tier.

## Final test-set results (Calibrated XGBoost, threshold = 0.25)

| Metric | Value |
|---|---|
| Precision (Churn) | 0.51 |
| Recall (Churn) | 0.81 |
| F1 (Churn) | 0.62 |
| ROC-AUC | 0.826 |
| PR-AUC | 0.641 |
| Brier Score | 0.142 |

At the lower, F1-optimized threshold the model catches most churners (81% recall) at the cost of more false positives (51% precision) — a trade-off that should be set deliberately based on the cost of a missed churner vs. an unnecessary retention offer, not left at an unexamined default.

## Requirements

```
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
```

## How to run

Open `Churn_Prediction_v2.ipynb` in Jupyter and run all cells top to bottom (it downloads the dataset directly from GitHub, no local data file needed).
