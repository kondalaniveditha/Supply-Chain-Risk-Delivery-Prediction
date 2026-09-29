# Supply Chain Risk Analytics and Delivery Time Prediction Using Machine Learning

**IBM SkillsBuild Data Analytics with AI Internship | Graduate-Level Project**  
**Author:** K. Niveditha Sai | **Year:** 2026

---

## Project Overview

An end-to-end supply chain analytics and machine learning system that predicts delivery time and late-delivery risk using only pre-delivery information. Built on 180,519 real shipment records from the DataCo Global supply chain dataset.

**Best Regression Model:** Random Forest — MAE = 0.9539, RMSE = 1.259 days, R² = 0.3929  
**Best Classification Model:** XGBoost — F1 = 0.6953, ROC-AUC = 0.7684

> These metrics are from the **leakage-corrected** pipeline (Order Status removed).

---

## Problem Statement

Given information available at the time of order placement, predict:
1. **Delivery time** (days) — regression
2. **Probability of late delivery** — binary classification

The system is designed as a realistic pre-delivery predictor. All post-delivery information is strictly excluded from model inputs.

---

## Data Leakage Prevention

> **Critical design constraint:** This project simulates a realistic pre-delivery prediction system.

The following columns are **strictly excluded** from all model inputs:

| Column | Reason |
|--------|--------|
| `Delivery Status` | Post-delivery outcome — leakage |
| `shipping date (DateOrders)` | Actual dispatch date — post-commitment info |
| `Order Status` | Post-fulfillment lifecycle state — CANCELED/SUSPECTED_FRAUD map 100% to Shipping canceled (Late_delivery_risk=0), direct target leakage |
| `Late_delivery_risk` | Classification target |
| `Days for shipping (real)` | Regression target |
| PII columns | Privacy protection |

---

## Dataset

**Name:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS  
**Source:** Constante, F., Silva, F., & Pereira, A. (2019). Mendeley Data.  
**DOI:** https://doi.org/10.17632/8gx2fvg2k6.5  
**Records:** 180,519 rows × 53 columns  
**Markets:** Africa, Europe, LATAM, Pacific Asia, USCA  

Place the dataset at:
```
data/raw/DataCoSupplyChainDataset.csv
```

---

## Technologies Used

| Category | Library | Version |
|----------|---------|---------|
| Data processing | pandas | 3.0.6 |
| Numerical | numpy | 2.5.3 |
| Visualisation | matplotlib, seaborn | 3.11.2, 0.13.2 |
| Machine learning | scikit-learn | 1.9.1 |
| Gradient boosting | xgboost | 3.4.1 |
| Explainability | shap | 0.52.0 |
| Dashboard | streamlit | 1.64.0 |
| Model persistence | joblib | 1.6.0 |
| Notebook | jupyter | 1.1.1 |
| Report generation | python-docx | 1.2.0 |
| Testing | pytest | 9.1.1 |

---

## Project Architecture

```
Raw Data (180,519 records)
       ↓
Data Preprocessing (src/data_preprocessing.py)
  • Remove leakage (Delivery Status, shipping date, Order Status), PII, identifiers
  • Parse dates, create datetime features
  • Chronological 80/20 train/test split
  • Label encode categoricals
       ↓
Feature Engineering (src/feature_engineering.py)
  • 31 final numeric features (after leakage correction)
       ↓
┌─────────────────────────┬────────────────────────────┐
│ Regression              │ Classification             │
│ Target: delivery days   │ Target: late delivery risk │
│ Models: RF, GBM, XGB,  │ Models: LR, RF, GBM, XGB  │
│         Linear, Baseline│                            │
└─────────────────────────┴────────────────────────────┘
       ↓
Model Evaluation (src/evaluation.py)
  • MAE, RMSE, R² | Accuracy, F1, ROC-AUC
       ↓
SHAP Explainability (src/explainability.py)
  • Global feature importance
  • SHAP summary plots
       ↓
Prediction Pipeline (src/prediction.py)
  • predict_single() → {days, probability, risk_category}
       ↓
Interactive Dashboard (dashboard/app.py)
  • Page 1: Executive Overview
  • Page 2: Delivery Performance
  • Page 3: Risk & Recommendations
```

---

## Repository Structure

```
Supply-Chain-Risk-Delivery-Prediction/
├── data/
│   ├── raw/                    # Original dataset (not committed to git)
│   └── processed/              # Preprocessed CSVs (generated)
│
├── notebooks/
│   └── Niveditha_SupplyChainRisk_DeliveryPrediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Full preprocessing pipeline
│   ├── feature_engineering.py  # Derived features
│   ├── regression_models.py    # Regression model definitions
│   ├── classification_models.py # Classification model definitions
│   ├── evaluation.py           # Metrics and comparison tables
│   ├── explainability.py       # SHAP utilities
│   └── prediction.py           # Inference pipeline
│
├── models/                     # Saved models and metadata
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard (3 pages)
│
├── reports/
│   ├── Niveditha_SupplyChain_ProjectReport.docx
│   └── figures/                # All visualisations
│
├── scripts/
│   ├── train_and_evaluate.py   # End-to-end training runner
│   ├── build_notebook.py       # Notebook generator
│   └── build_report.py         # DOCX report generator
│
├── tests/
│   ├── test_preprocessing.py   # 24 preprocessing tests
│   └── test_models.py          # 16 model tests
│
├── docs/
│   └── DATA_DICTIONARY.md      # Full column documentation
│
├── requirements.txt
├── AGENTS.md
└── README.md
```

---

## Machine Learning

### Regression (Delivery Time)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Random Forest** ✓ | **0.9539** | **1.2590** | **0.3929** |
| Gradient Boosting | 0.9682 | 1.2618 | 0.3902 |
| XGBoost | 0.9745 | 1.2634 | 0.3887 |
| Linear Regression | 0.9840 | 1.2643 | 0.3878 |
| Baseline (Mean) | 1.4186 | 1.6158 | 0.0000 |

### Classification (Late Delivery Risk)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|--------|-----|---------|
| **XGBoost** ✓ (selected by F1) | **0.7160** | **0.8404** | **0.5930** | **0.6953** | **0.7684** |
| Gradient Boosting | 0.7161 | 0.8407 | 0.5928 | 0.6953 | 0.7669 |
| Random Forest | 0.7178 | 0.8500 | 0.5873 | 0.6947 | 0.7685 |
| Logistic Regression | 0.6936 | 0.8424 | 0.5404 | 0.6584 | 0.7394 |

All metrics are from the **leakage-free pipeline** (Order Status excluded). Train/test split is **chronological** (80% train, 20% test).

Model selection: XGBoost is selected by the highest F1 score (primary metric). XGBoost and Gradient Boosting both report F1 = 0.6953 at the displayed precision; XGBoost appears first in the F1-sorted comparison. Note: Random Forest has the highest ROC-AUC at 0.7685.

---

## Explainable AI

SHAP (TreeExplainer) values are computed on a sample of the test set. Top factors differ by task:

**Regression (Random Forest) — top SHAP factors (mean |SHAP value|):**
1. **Days for shipment (scheduled)** — strongest factor associated with predicted delivery days
2. **Shipping Mode** — second strongest factor
3. **order_hour** — contributes at considerably lower magnitude
4. **Order Region** and **Type** — further contributors at lower magnitude

**Classification (XGBoost) — top SHAP factors (mean |SHAP value|):**
1. **Days for shipment (scheduled)** — dominant factor associated with predicted late-delivery risk
2. **Shipping Mode** — second strongest factor
3. **Type** and **order_hour** — contribute at lower magnitude
4. **Order Region**, **order_day**, **order_dayofweek** — modest contributors

> SHAP values indicate the strength of association between a feature and model output. They do not establish causation.

---

## Key Findings

1. **54.8% of shipments are late** — systemic, not isolated
2. Standard Class shipping drives the majority of late deliveries by volume
3. The XGBoost classifier achieves ROC-AUC ≈ 0.768 using pre-delivery info only (leakage-free); Random Forest has the highest ROC-AUC at 0.7685
4. Scheduled delivery days is the strongest pre-delivery predictor (top SHAP factor in both tasks)
5. Same Day delivery has the lowest late rate but is underutilised (~5% of shipments)

---

## Business Recommendations

1. Deploy the predictive model at order confirmation to assign risk scores in real-time
2. Review Standard Class SLA performance with logistics partners; explore carrier diversification where evidence supports it
3. Implement risk-tiered intervention: HIGH → dispatch alert; MEDIUM → customer notification
4. Use model predictions to flag cases where predicted delivery days exceed scheduled days
5. Evaluate targeted expedited-shipping eligibility for high-value Corporate customers based on predicted risk and cost-benefit analysis

---

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.13.4)
- Windows/Linux/macOS

### Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the dataset from Mendeley Data (DOI: 10.17632/8gx2fvg2k6.5) and place at:
```
data/raw/DataCoSupplyChainDataset.csv
```

---

## How to Run

### 1. Train all models and generate figures
```bash
venv\Scripts\python scripts/train_and_evaluate.py
```
This runs preprocessing, trains all models, generates all figures, and saves artifacts.

### 2. Run the interactive dashboard
```bash
venv\Scripts\streamlit run dashboard/app.py
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook notebooks/Niveditha_SupplyChainRisk_DeliveryPrediction.ipynb
```

### 4. Run tests
```bash
venv\Scripts\python -m pytest tests/ -v
```

### 5. Generate the DOCX report
```bash
venv\Scripts\python scripts/build_report.py
```

### 6. Regenerate notebook
```bash
venv\Scripts\python scripts/build_notebook.py
```

---

## Dashboard Instructions

The dashboard has three pages accessible via the sidebar:

- **📊 Executive Overview** — KPIs, delivery trends, market/mode performance
- **📦 Delivery Performance** — drivers of delivery time and delay rates
- **⚠️ Risk & Recommendations** — model results, SHAP insights, business actions

Filters in the sidebar (Market, Region, Ship Mode, Customer Segment) apply to all dashboard pages.

---

## Model Artifacts

Saved in `models/`:
- `best_regression_model.joblib` — Random Forest regressor
- `best_regression_model_scaler.joblib` — StandardScaler
- `best_regression_model_features.joblib` — 31 feature names
- `best_classification_model.joblib` — XGBoost classifier
- `best_classification_model_scaler.joblib` — StandardScaler
- `best_classification_model_features.joblib` — 31 feature names
- `training_summary.json` — all metrics

To reproduce results exactly:
1. Place raw dataset in `data/raw/`
2. Run `venv\Scripts\python scripts/train_and_evaluate.py`
3. All random states are fixed at seed=42
4. The chronological split is deterministic (no shuffling)

---

## Limitations

- Dataset covers 2015–2018; patterns may differ in current supply chains
- Regression R² ≈ 0.39 — inherent randomness in logistics limits predictability
- SHAP values indicate association, not causation
- Optimal probability thresholds (LOW/MEDIUM/HIGH) require operational cost analysis
- Static model; production use requires periodic retraining

---

## Future Scope

- Real-time API integration with order management systems
- Carrier-level performance modelling
- Weather and geopolitical risk features
- Time-series demand forecasting
- Automated retraining with data drift detection

---

## Author

**K. Niveditha Sai**  
IBM SkillsBuild Data Analytics with AI Internship  
Programme: Graduate-Level Data Analytics | Year: 2026

---

## Citation

If you use this dataset:

```
Constante, F., Silva, F., & Pereira, A. (2019).
DataCo Smart Supply Chain for Big Data Analysis.
Mendeley Data, V5. https://doi.org/10.17632/8gx2fvg2k6.5
```
