# Supply Chain Risk Analytics and Delivery Time Prediction Using Machine Learning

**IBM SkillsBuild Data Analytics with AI Internship | Graduate-Level Project**  
**Author:** Niveditha

---

## Project Overview

An end-to-end supply chain analytics and machine learning system that predicts delivery time and late-delivery risk using only pre-delivery information. Built on 180,519 real shipment records from the DataCo Global supply chain dataset.

**Best Regression Model:** Random Forest — RMSE = 1.259 days, R² = 0.393  
**Best Classification Model:** Gradient Boosting — F1 = 0.710, ROC-AUC = 0.800

---

## Problem Statement

Given information available at the time of order placement, predict:
1. **Delivery time** (days) — regression
2. **Probability of late delivery** — binary classification

The system is designed as a realistic pre-delivery predictor. All post-delivery information is strictly excluded from model inputs.

---

## Objectives

1. Audit the DataCo supply chain dataset (180,519 records, 53 columns)
2. Build a reproducible preprocessing pipeline with strict leakage prevention
3. Train and compare regression and classification models
4. Apply SHAP explainability to identify key predictive factors
5. Provide an interactive executive dashboard
6. Translate findings into actionable business recommendations

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
data/raw/DescriptionDataCoSupplyChain.csv
```

---

## Data Leakage Prevention

> **Critical design constraint:** This project simulates a realistic pre-delivery prediction system.

The following columns are **strictly excluded** from all model inputs:

| Column | Reason |
|--------|--------|
| `Delivery Status` | Post-delivery outcome — leakage |
| `shipping date (DateOrders)` | Actual dispatch date — post-commitment info |
| `Late_delivery_risk` | Classification target |
| `Days for shipping (real)` | Regression target |
| PII columns | Privacy protection |

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
  • Remove leakage, PII, identifiers
  • Parse dates, create datetime features
  • Chronological 80/20 train/test split
  • Label encode categoricals
       ↓
Feature Engineering (src/feature_engineering.py)
  • 32 final numeric features
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
| **Random Forest** ✓ | **0.9545** | **1.2592** | **0.3928** |
| XGBoost | 0.9714 | 1.2624 | 0.3896 |
| Gradient Boosting | 0.9692 | 1.2628 | 0.3893 |
| Linear Regression | 0.9843 | 1.2644 | 0.3877 |
| Baseline (Mean) | 1.4186 | 1.6158 | 0.0000 |

### Classification (Late Delivery Risk)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|--------|-----|---------|
| **Gradient Boosting** ✓ | **0.7356** | **0.8868** | **0.5917** | **0.7098** | **0.8004** |
| XGBoost | 0.7363 | 0.8914 | 0.5894 | 0.7096 | 0.7980 |
| Random Forest | 0.7347 | 0.8898 | 0.5873 | 0.7076 | 0.7885 |
| Logistic Regression | 0.6983 | 0.8519 | 0.5423 | 0.6627 | 0.7377 |

---

## Explainable AI

SHAP (TreeExplainer) identifies the following as the strongest predictive factors:

1. **Days for shipment (scheduled)** — scheduling estimate is the primary signal
2. **Shipping Mode** — Same Day shows lower risk; Standard Class higher
3. **Order Region** — regional logistics infrastructure differences
4. **Financial variables** — Sales, Order Item Total (moderate importance)
5. **Temporal features** — month, day of week (seasonal patterns)

> Note: SHAP values show feature-prediction associations, not causation.

---

## Key Findings

1. **54.8% of shipments are late** — systemic, not isolated
2. Standard Class shipping drives the majority of late deliveries by volume
3. The Gradient Boosting classifier achieves ROC-AUC = 0.800 using pre-delivery info only
4. Scheduled delivery days is the strongest pre-delivery predictor
5. Same Day delivery has the lowest late rate but is underutilised (~5% of shipments)

---

## Business Recommendations

1. Deploy the predictive model at order confirmation to assign risk scores in real-time
2. Revise Standard Class SLA agreements with logistics partners
3. Implement risk-tiered intervention: HIGH → dispatch alert; MEDIUM → customer notification
4. Recalibrate scheduling estimates using model predictions
5. Automatic mode upgrades for high-value Corporate segment orders

---

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.13.4)
- Windows/Linux/macOS

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/[username]/supply-chain-risk-delivery-prediction.git
cd supply-chain-risk-delivery-prediction

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

Download the dataset from Mendeley Data (DOI: 10.17632/8gx2fvg2k6.5) and place:
```
data/raw/DataCoSupplyChainDataset.csv
data/raw/DescriptionDataCoSupplyChain.csv
```

---

## How to Run

### 1. Train all models and generate figures
```bash
python scripts/train_and_evaluate.py
```
This runs preprocessing, trains all models, generates all figures, and saves artifacts.

### 2. Run the interactive dashboard
```bash
streamlit run dashboard/app.py
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook notebooks/Niveditha_SupplyChainRisk_DeliveryPrediction.ipynb
```

### 4. Run tests
```bash
python -m pytest tests/ -v
```

### 5. Generate the DOCX report
```bash
python scripts/build_report.py
```

### 6. Single test
```bash
python -m pytest tests/test_preprocessing.py::TestDataLoading::test_load_returns_dataframe -v
python -m pytest tests/test_models.py::TestPredictionPipeline::test_risk_category_assignment -v
```

---

## Dashboard Instructions

The dashboard has three pages accessible via the sidebar:

- **📊 Executive Overview** — KPIs, delivery trends, market/mode performance
- **📦 Delivery Performance** — drivers of delivery time and delay rates
- **⚠️ Risk & Recommendations** — model results, SHAP insights, business actions

Filters in the sidebar (Market, Region, Ship Mode, Customer Segment) apply to all dashboard pages.

---

## Model Reproduction

To reproduce results exactly:
1. Place raw dataset in `data/raw/`
2. Run `python scripts/train_and_evaluate.py`
3. All random states are fixed at seed=42
4. The chronological split is deterministic (no shuffling)

Saved artifacts in `models/`:
- `best_regression_model.joblib` — Random Forest regressor
- `best_regression_model_scaler.joblib` — StandardScaler
- `best_regression_model_features.joblib` — feature name list
- `best_classification_model.joblib` — Gradient Boosting classifier
- `best_classification_model_scaler.joblib` — StandardScaler
- `best_classification_model_features.joblib` — feature name list
- `training_summary.json` — all metrics

---

## Limitations

- Dataset covers 2015–2018; patterns may differ in current supply chains
- Regression R² = 0.39 — inherent randomness in logistics limits predictability
- SHAP shows association, not causation
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

**Niveditha**  
IBM SkillsBuild Data Analytics with AI Internship  
Programme: Graduate-Level Data Analytics

---

## Citation

If you use this project or dataset:

```
Constante, F., Silva, F., & Pereira, A. (2019).
DataCo Smart Supply Chain for Big Data Analysis.
Mendeley Data, V5. https://doi.org/10.17632/8gx2fvg2k6.5
```
