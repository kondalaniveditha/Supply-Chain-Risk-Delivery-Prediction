"""
scripts/build_notebook.py
==========================
Generates the complete Jupyter notebook programmatically.
Run from project root:
    venv/Scripts/python scripts/build_notebook.py
"""

import json
import os

NB_PATH = "notebooks/Niveditha_SupplyChainRisk_DeliveryPrediction.ipynb"
os.makedirs("notebooks", exist_ok=True)


def md(text):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }


def code(lines, outputs=None):
    """Create a code cell."""
    if isinstance(lines, str):
        lines = [lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": lines,
    }


cells = []

# ============================================================
# SECTION 1 — TITLE AND ABSTRACT
# ============================================================
cells.append(md("""# Supply Chain Risk Analytics and Delivery Time Prediction Using Machine Learning

**Author:** Niveditha  
**Programme:** IBM SkillsBuild Data Analytics with AI Internship  
**Dataset:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS (DataCo Global, 180,519 records)  
**Date:** 2024

---

## Abstract

This project develops an end-to-end machine learning system for supply chain delivery analytics. Using 180,519 historical shipment records from the DataCo Global supply chain dataset, the system (1) predicts actual delivery time in days via regression, and (2) predicts the probability of late delivery via binary classification. A strict data-leakage prevention policy ensures that only information available before delivery is used as model input. The best regression model (Random Forest) achieves RMSE = 1.259 days and R² = 0.393. The best classification model (Gradient Boosting) achieves F1 = 0.710 and ROC-AUC = 0.800. Model explanations using SHAP identify scheduled delivery days, shipping mode, and order region as the most important predictive factors. Findings are translated into actionable business recommendations via an interactive Streamlit dashboard.

---

## Table of Contents

1. [Problem Statement & Objectives](#problem)
2. [Business Context](#business)
3. [Dataset Overview](#dataset)
4. [Data Dictionary](#dictionary)
5. [Data Quality Analysis](#quality)
6. [Data Cleaning & Preprocessing](#cleaning)
7. [Exploratory Data Analysis](#eda)
8. [Feature Engineering](#features)
9. [Machine Learning Methodology](#methodology)
10. [Delivery Time Prediction (Regression)](#regression)
11. [Delivery Risk Prediction (Classification)](#classification)
12. [Model Evaluation & Comparison](#evaluation)
13. [Explainable AI (SHAP)](#xai)
14. [Business Insights](#insights)
15. [Risk Analysis](#risk)
16. [Recommendations](#recommendations)
17. [Conclusion](#conclusion)
"""))

# SECTION 2 — PROBLEM STATEMENT
cells.append(md("""## 1. Problem Statement & Objectives <a id="problem"></a>

### Problem Statement

Supply chain delivery performance is a critical operational challenge. Late deliveries directly impact customer satisfaction, operational costs, and competitive positioning. However, traditional reactive approaches — investigating a delay *after* it occurs — offer limited business value. The goal of this project is to build a **predictive** system that identifies shipments at risk of late delivery *before* they are dispatched, enabling proactive intervention.

### Data Leakage Prevention Policy

> **CRITICAL:** This project is designed as a realistic **pre-delivery** prediction system. The following columns are **strictly excluded** from all model inputs because they would only become known after the shipment outcome:
> - `Delivery Status` — direct consequence of the delivery outcome
> - `shipping date (DateOrders)` — actual dispatch date (post-commitment information)  
> - `Late_delivery_risk` — the classification target itself
> - `Days for shipping (real)` — the regression target itself
> - All PII columns

### Objectives

1. Audit and understand the DataCo supply chain dataset.
2. Implement a reproducible preprocessing pipeline with leakage prevention.
3. Conduct exploratory data analysis to identify patterns in delivery performance.
4. Build and compare regression models for delivery time prediction.
5. Build and compare classification models for late-delivery risk prediction.
6. Apply SHAP-based explainability to interpret model predictions.
7. Translate findings into actionable business recommendations.
8. Deploy an interactive executive dashboard.
"""))

# SECTION 3 — BUSINESS CONTEXT
cells.append(md("""## 2. Business Context <a id="business"></a>

Supply chain management is a multi-billion-dollar operational domain where delivery performance directly affects:

- **Customer Retention** — late deliveries are a leading cause of customer churn in e-commerce and B2B supply.
- **Operational Costs** — expedited rerouting and compensation for late deliveries increase logistics costs.
- **Competitive Positioning** — delivery reliability is a primary differentiation factor in modern supply chains.
- **Inventory Management** — predictable delivery windows reduce safety-stock requirements.

The DataCo dataset covers shipments across five global markets (Africa, Europe, LATAM, Pacific Asia, USCA) from 2015–2018, providing a realistic representation of a multi-geography supply chain operation.

**Business Question:** *Given information available at the time of order placement, what is the expected delivery time and what is the probability that this shipment will be late?*
"""))

# SECTION 4 — SETUP
cells.append(md("""## 3. Setup and Imports"""))
cells.append(code("""import os, sys, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Add project src to path
sys.path.insert(0, os.path.abspath('..'))

from src.data_preprocessing import (
    load_raw_data, report_data_quality, parse_dates,
    create_datetime_features, drop_leakage_and_pii,
    handle_missing_values, encode_categoricals, build_feature_matrix,
    chronological_split, run_full_pipeline,
    LEAKAGE_COLS, PII_COLS, TARGET_REGRESSION, TARGET_CLASSIFICATION,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from src.feature_engineering import add_all_features, get_final_feature_columns
from src.regression_models import train_regression_models, save_regression_artifacts
from src.classification_models import train_classification_models, save_classification_artifacts
from src.evaluation import (
    compare_regression_models, compare_classification_models,
    select_best_regression_model, select_best_classification_model,
    get_confusion_matrix
)
from src.explainability import compute_shap_values, get_top_shap_features

sns.set_theme(style='whitegrid', palette='muted')
FIGURES_DIR = '../reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
print('Environment ready.')"""))

# SECTION 5 — DATASET OVERVIEW
cells.append(md("""## 4. Dataset Overview <a id="dataset"></a>

**Source:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS  
**Reference:** Constante, F., Silva, F., & Pereira, A. (2019). DataCo Smart Supply Chain for Big Data Analysis. Mendeley Data.  
**File:** `data/raw/DataCoSupplyChainDataset.csv`  
**Encoding:** Latin-1
"""))
cells.append(code("""df_raw = load_raw_data('../data/raw/DataCoSupplyChainDataset.csv')
print(f"Dataset shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"\\nColumn names:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col}")"""))

cells.append(code("""print("\\nBasic statistics for key numerical variables:")
key_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)',
            'Sales', 'Order Item Quantity', 'Product Price']
df_raw[key_cols].describe().round(3)"""))

cells.append(code("""print("Target variable distributions:")
print(f"\\nDays for shipping (real):")
print(df_raw['Days for shipping (real)'].value_counts().sort_index())
print(f"\\nLate_delivery_risk:")
print(df_raw['Late_delivery_risk'].value_counts())
late_pct = df_raw['Late_delivery_risk'].mean() * 100
print(f"\\nLate delivery rate: {late_pct:.1f}%")"""))

# SECTION 6 — DATA QUALITY
cells.append(md("""## 5. Data Quality Analysis <a id="quality"></a>

A systematic quality audit is performed before any cleaning. This identifies missing values, data types, and potential issues.
"""))
cells.append(code("""quality = report_data_quality(df_raw)
print("Data Quality Report:")
print(quality.to_string())"""))

cells.append(code("""print("Missing value summary:")
missing = quality[quality['missing_count'] > 0][['dtype', 'missing_count', 'missing_pct']]
print(missing)
print(f"\\nDuplicate rows: {df_raw.duplicated().sum()}")"""))

# SECTION 7 — DATA CLEANING
cells.append(md("""## 6. Data Cleaning & Preprocessing <a id="cleaning"></a>

The preprocessing pipeline performs the following steps in order:
1. **Parse dates** — convert order date string to datetime
2. **Create datetime features** — extract year, month, quarter, day, dayofweek, hour
3. **Remove leakage** — drop Delivery Status, shipping date (post-delivery info)
4. **Remove PII** — drop customer email, name, password, address
5. **Remove identifiers** — drop meaningless ID columns
6. **Remove sparse columns** — drop columns with >80% missing or near-zero signal
7. **Impute missing values** — median for numeric, 'Unknown' for categorical
8. **Encode categoricals** — LabelEncoder (fit on train only)
9. **Chronological split** — 80% train (earlier), 20% test (later)

> All steps are applied to a **copy** of the raw data. Raw data is never modified.
"""))
cells.append(code("""print("Running full preprocessing pipeline...")
pipeline_result = run_full_pipeline(
    raw_path='../data/raw/DataCoSupplyChainDataset.csv',
    processed_path='../data/processed/supply_chain_processed.csv'
)

df_full     = pipeline_result['df_full']
df_train    = pipeline_result['df_train']
df_test     = pipeline_result['df_test']
encoders    = pipeline_result['encoders']

print(f"\\nPreprocessed full dataset: {len(df_full):,} rows")
print(f"Train split: {len(df_train):,} rows")
print(f"Test  split: {len(df_test):,} rows")
print(f"\\nRemaining columns ({len(df_full.columns)}):")
print(df_full.columns.tolist())"""))

cells.append(code("""# Verify no leakage columns remain
print("Leakage column check:")
for col in LEAKAGE_COLS + ['shipping date (DateOrders)']:
    status = '✗ PRESENT (ERROR)' if col in df_full.columns else '✓ Removed'
    print(f"  {col}: {status}")

print("\\nPII column check:")
for col in PII_COLS:
    status = '✗ PRESENT (ERROR)' if col in df_full.columns else '✓ Removed'
    print(f"  {col}: {status}")"""))

# SECTION 8 — EDA
cells.append(md("""## 7. Exploratory Data Analysis <a id="eda"></a>

### Key EDA Questions
1. What is the distribution of actual delivery times?
2. What percentage of shipments are late?
3. Which shipping modes perform differently?
4. Which markets/regions have higher delay rates?
5. Are there temporal patterns in delivery performance?
6. Which numerical variables correlate with delivery time?
"""))

cells.append(code("""# Q1: Delivery time distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dist = df_raw['Days for shipping (real)'].value_counts().sort_index()
colors = ['#ef4444' if i >= 4 else '#3b82d4' for i in dist.index]
axes[0].bar(dist.index, dist.values, color=colors, edgecolor='white')
axes[0].set_xlabel('Actual Delivery Days', fontsize=12)
axes[0].set_ylabel('Number of Shipments', fontsize=12)
axes[0].set_title('Distribution of Actual Delivery Days', fontsize=13, fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Scheduled vs actual
sched_dist = df_raw['Days for shipment (scheduled)'].value_counts().sort_index()
axes[1].bar(sched_dist.index, sched_dist.values, color='#7c5cd8', edgecolor='white')
axes[1].set_xlabel('Scheduled Delivery Days', fontsize=12)
axes[1].set_ylabel('Number of Shipments', fontsize=12)
axes[1].set_title('Distribution of Scheduled Delivery Days', fontsize=13, fontweight='bold')

plt.suptitle('Actual vs Scheduled Delivery Days Distribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_delivery_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean actual days:     {df_raw['Days for shipping (real)'].mean():.2f}")
print(f"Median actual days:   {df_raw['Days for shipping (real)'].median():.0f}")
print(f"Mean scheduled days:  {df_raw['Days for shipment (scheduled)'].mean():.2f}")"""))

cells.append(code("""# Q2 & Q3: Late delivery by shipping mode
ship_perf = df_raw.groupby('Shipping Mode').agg(
    total=('Days for shipping (real)', 'count'),
    avg_days=('Days for shipping (real)', 'mean'),
    late_rate=('Late_delivery_risk', 'mean')
).reset_index().sort_values('late_rate', ascending=False)

print("Delivery performance by shipping mode:")
print(ship_perf.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(ship_perf.sort_values('avg_days')['Shipping Mode'],
             ship_perf.sort_values('avg_days')['avg_days'], color='#3b82d4')
axes[0].set_xlabel('Average Delivery Days', fontsize=12)
axes[0].set_title('Avg Delivery Days by Shipping Mode', fontsize=12, fontweight='bold')

axes[1].barh(ship_perf.sort_values('late_rate')['Shipping Mode'],
             ship_perf.sort_values('late_rate')['late_rate'] * 100, color='#f59e0b')
axes[1].set_xlabel('Late Delivery Rate (%)', fontsize=12)
axes[1].set_title('Late Rate by Shipping Mode', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_shipping_mode_perf.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# Q4: Market and regional performance
mkt_perf = df_raw.groupby('Market').agg(
    avg_days=('Days for shipping (real)', 'mean'),
    late_rate=('Late_delivery_risk', 'mean'),
    count=('Days for shipping (real)', 'count')
).reset_index().sort_values('late_rate', ascending=False)

print("Delivery performance by market:")
print(mkt_perf.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(mkt_perf.sort_values('late_rate')['Market'],
             mkt_perf.sort_values('late_rate')['late_rate'] * 100,
             color=['#ef4444' if r > 0.55 else '#3b82d4' for r in mkt_perf.sort_values('late_rate')['late_rate']])
axes[0].set_xlabel('Late Delivery Rate (%)', fontsize=12)
axes[0].set_title('Late Rate by Market', fontsize=12, fontweight='bold')

axes[1].barh(mkt_perf.sort_values('avg_days')['Market'],
             mkt_perf.sort_values('avg_days')['avg_days'], color='#7c5cd8')
axes[1].set_xlabel('Average Delivery Days', fontsize=12)
axes[1].set_title('Avg Delivery Days by Market', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_market_perf.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# Q5: Temporal trend
df_raw['order_date'] = pd.to_datetime(df_raw['order date (DateOrders)'], format='%m/%d/%Y %H:%M', errors='coerce')
monthly = df_raw.set_index('order_date').resample('ME').agg(
    avg_days=('Days for shipping (real)', 'mean'),
    late_rate=('Late_delivery_risk', 'mean'),
    volume=('Days for shipping (real)', 'count')
)

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
axes[0].plot(monthly.index, monthly['avg_days'], '#3b82d4', linewidth=2)
axes[0].set_ylabel('Avg Delivery Days', fontsize=11)
axes[0].set_title('Monthly Supply Chain Performance', fontsize=13, fontweight='bold')
axes[0].grid(linestyle='--', alpha=0.4)

axes[1].plot(monthly.index, monthly['late_rate'] * 100, '#ef4444', linewidth=2)
axes[1].axhline(df_raw['Late_delivery_risk'].mean() * 100, color='black',
                linestyle='--', linewidth=1, label=f"Avg {df_raw['Late_delivery_risk'].mean()*100:.1f}%")
axes[1].set_ylabel('Late Rate (%)', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(linestyle='--', alpha=0.4)

axes[2].bar(monthly.index, monthly['volume'], color='#7c5cd8', width=25, alpha=0.7)
axes[2].set_ylabel('Shipment Volume', fontsize=11)
axes[2].set_xlabel('Month', fontsize=11)
axes[2].grid(linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_monthly_trend.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# Q6: Correlation of numerical features with targets
num_corr_cols = NUMERICAL_FEATURES + ['Days for shipping (real)', 'Late_delivery_risk']
corr_matrix = df_raw[num_corr_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
import numpy as np
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, annot_kws={'size': 8}, linewidths=0.3,
            vmin=-1, vmax=1)
ax.set_title('Pearson Correlation Heatmap — Numerical Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nCorrelation with Days for shipping (real):")
corr_target = corr_matrix['Days for shipping (real)'].drop('Days for shipping (real)').sort_values()
print(corr_target.to_string())"""))

# SECTION 9 — FEATURE ENGINEERING
cells.append(md("""## 8. Feature Engineering <a id="features"></a>

Beyond the basic preprocessing, the following additional features are created using only pre-delivery information:

| Feature | Description | Source |
|---------|-------------|--------|
| `order_year/month/quarter/day/dayofweek/hour` | Date components | Order date |
| `is_expedited_schedule` | 1 if scheduled days ≤ 2 | Scheduled days |
| `profit_margin_ratio` | Order Profit / Order Item Total | Order financials |
| `has_discount` | 1 if item discount > 0 | Item discount |
| `is_high_value` | 1 if Product Price ≥ $200 | Product price |
| `is_weekend_order` | 1 if order placed on Sat/Sun | Order dayofweek |
"""))
cells.append(code("""# Apply feature engineering to train/test splits
X_train_raw, X_test_raw, y_reg_train, y_reg_test, y_clf_train, y_clf_test = None, None, None, None, None, None

df_train_fe = add_all_features(df_train)
df_test_fe  = add_all_features(df_test)

X_train, y_reg_train, y_clf_train = build_feature_matrix(df_train_fe)
X_test,  y_reg_test,  y_clf_test  = build_feature_matrix(df_test_fe)

# Keep only numeric columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train = X_train[num_cols]
X_test  = X_test.reindex(columns=num_cols, fill_value=0)

print(f"Final feature set ({len(num_cols)} features):")
for i, col in enumerate(num_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Late rate (train): {y_clf_train.mean()*100:.1f}%")
print(f"Late rate (test):  {y_clf_test.mean()*100:.1f}%")"""))

# SECTION 10 — MACHINE LEARNING METHODOLOGY
cells.append(md("""## 9. Machine Learning Methodology <a id="methodology"></a>

### Train/Test Split Strategy

A **chronological split** is used: the earliest 80% of orders form the training set, and the latest 20% form the test set. This prevents data leakage from future shipments into the training data, reflecting a realistic deployment scenario where a model trained on historical data predicts the outcome of future shipments.

### Models Evaluated

**Regression (predicting delivery days):**
- Baseline (Mean predictor)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Classification (predicting late delivery risk):**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Class Imbalance

The classification target has ~55% late and ~45% on-time deliveries — a mild imbalance. `class_weight='balanced'` is used in applicable models to prevent bias toward the majority class.

### Feature Scaling

`StandardScaler` is applied to all features before model training. The scaler is fit on training data only and applied to test data to prevent data leakage through scaling statistics.
"""))

# SECTION 11 — REGRESSION
cells.append(md("""## 10. Delivery Time Prediction (Regression) <a id="regression"></a>

**Target:** `Days for shipping (real)` — the actual number of days taken to deliver the shipment.
"""))
cells.append(code("""from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Training regression models...")
reg_models, reg_scaler = train_regression_models(X_train, y_reg_train)
X_test_reg = reg_scaler.transform(X_test)

reg_comparison = compare_regression_models(reg_models, X_test_reg, y_reg_test)
print("\\nRegression Model Comparison:")
print(reg_comparison.to_string(index=False))"""))

cells.append(code("""# Load pre-computed comparison if available
import json
if os.path.exists('../models/training_summary.json'):
    with open('../models/training_summary.json') as f:
        summary = json.load(f)
    reg_best = summary['regression']['best_model']
    print(f"Best Regression Model: {reg_best['model']}")
    print(f"  MAE  = {reg_best['MAE']}")
    print(f"  RMSE = {reg_best['RMSE']}")
    print(f"  R²   = {reg_best['R2']}")"""))

cells.append(code("""best_reg_name, best_reg_model = select_best_regression_model(reg_comparison, reg_models)
save_regression_artifacts(best_reg_model, reg_scaler, num_cols, '../models')
print(f"Best model saved: {best_reg_name}")

# Visualise actual vs predicted
y_reg_pred = best_reg_model.predict(X_test_reg)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_reg_test, y_reg_pred, alpha=0.1, s=8, color='#3b82d4')
mn, mx = min(y_reg_test.min(), y_reg_pred.min()), max(y_reg_test.max(), y_reg_pred.max())
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
axes[0].set_xlabel('Actual Delivery Days', fontsize=12)
axes[0].set_ylabel('Predicted Delivery Days', fontsize=12)
axes[0].set_title(f'{best_reg_name} — Actual vs Predicted', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

residuals = y_reg_test - y_reg_pred
axes[1].hist(residuals, bins=30, color='#7c5cd8', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Residual (Actual − Predicted)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')

plt.suptitle('Regression Model Diagnostics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/regression_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Residual mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")"""))

# SECTION 12 — CLASSIFICATION
cells.append(md("""## 11. Delivery Risk Prediction (Classification) <a id="classification"></a>

**Target:** `Late_delivery_risk` — binary flag (0=on time, 1=late delivery).
"""))
cells.append(code("""print("Training classification models...")
clf_models, clf_scaler = train_classification_models(X_train, y_clf_train)
X_test_clf = clf_scaler.transform(X_test)

clf_comparison = compare_classification_models(clf_models, X_test_clf, y_clf_test)
print("\\nClassification Model Comparison:")
print(clf_comparison.to_string(index=False))"""))

cells.append(code("""best_clf_name, best_clf_model = select_best_classification_model(clf_comparison, clf_models)
save_classification_artifacts(best_clf_model, clf_scaler, num_cols, '../models')
print(f"Best model saved: {best_clf_name}")"""))

cells.append(code("""from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
y_clf_pred = best_clf_model.predict(X_test_clf)
cm = get_confusion_matrix(y_clf_test, y_clf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['On Time', 'Late'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix — {best_clf_name}', fontsize=12, fontweight='bold')

# ROC curves
for name, model in clf_models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_clf)[:, 1]
        fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curves — All Classifiers', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/classification_roc_confusion.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# SECTION 13 — MODEL COMPARISON
cells.append(md("""## 12. Model Evaluation & Comparison <a id="evaluation"></a>

### Regression Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | 0.9545 | 1.2592 | 0.3928 |
| XGBoost | 0.9714 | 1.2624 | 0.3896 |
| Gradient Boosting | 0.9692 | 1.2628 | 0.3893 |
| Linear Regression | 0.9843 | 1.2644 | 0.3877 |
| Baseline (Mean) | 1.4186 | 1.6158 | 0.0000 |

**Selected model: Random Forest** (lowest RMSE, highest R²)

### Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|--------|-----|---------|
| Gradient Boosting | 0.7356 | 0.8868 | 0.5917 | 0.7098 | 0.8004 |
| XGBoost | 0.7363 | 0.8914 | 0.5894 | 0.7096 | 0.7980 |
| Random Forest | 0.7347 | 0.8898 | 0.5873 | 0.7076 | 0.7885 |
| Logistic Regression | 0.6983 | 0.8519 | 0.5423 | 0.6627 | 0.7377 |

**Selected model: Gradient Boosting** (best F1, best ROC-AUC)

### Interpretation Notes

- The regression R² of ~0.39 indicates the model explains about 39% of delivery time variance using pre-delivery features only. The dataset has inherent randomness (same shipping mode, same route, different outcomes), which limits achievable R².
- The classifier ROC-AUC of 0.800 means it correctly ranks 80% of late deliveries above on-time ones — significantly better than chance (0.5) and competitive with supervised learning baselines on this dataset.
- High Precision (0.887) with moderate Recall (0.591) means: when the model predicts a shipment will be late, it is correct 88.7% of the time, but it misses some late deliveries.
"""))

# SECTION 14 — XAI
cells.append(md("""## 13. Explainable AI (SHAP) <a id="xai"></a>

SHAP (SHapley Additive exPlanations) values quantify how much each feature contributes to individual model predictions. The analysis below provides global (dataset-level) and local (individual prediction) explanations.

> **Important:** SHAP values indicate the strength of association between a feature and model output. They do not establish causation. Language such as "associated with predictions" or "important predictive factor" is used throughout.
"""))
cells.append(code("""import shap, os
shap.initjs()

X_train_reg_scaled = reg_scaler.transform(X_train)
X_test_reg_scaled  = reg_scaler.transform(X_test)
X_train_clf_scaled = clf_scaler.transform(X_train)
X_test_clf_scaled  = clf_scaler.transform(X_test)

# SHAP for best regression model
print("Computing SHAP values for regression model...")
shap_vals_reg, X_sample_reg = compute_shap_values(
    best_reg_model, X_test_reg_scaled, X_train_reg_scaled, sample_size=1000
)

top_reg = get_top_shap_features(shap_vals_reg, num_cols, top_n=10)
print("\\nTop 10 features (regression - by mean |SHAP|):")
print(top_reg.to_string(index=False))"""))

cells.append(code("""# Plot SHAP bar for regression
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(9, 6))
mean_abs = np.abs(shap_vals_reg).mean(axis=0)
imp_df = top_reg.sort_values('mean_abs_shap')
ax.barh(imp_df['feature'], imp_df['mean_abs_shap'], color='#3b82d4')
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.set_title(f'SHAP Global Importance — {best_reg_name} (Delivery Days)', fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/notebook_shap_regression_bar.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(code("""# SHAP for best classification model
print("Computing SHAP values for classification model...")
shap_vals_clf, X_sample_clf = compute_shap_values(
    best_clf_model, X_test_clf_scaled, X_train_clf_scaled, sample_size=1000
)

top_clf = get_top_shap_features(shap_vals_clf, num_cols, top_n=10)
print("\\nTop 10 features (classification - by mean |SHAP|):")
print(top_clf.to_string(index=False))"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 6))
imp_clf = top_clf.sort_values('mean_abs_shap')
ax.barh(imp_clf['feature'], imp_clf['mean_abs_shap'], color='#ef4444')
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.set_title(f'SHAP Global Importance — {best_clf_name} (Late Delivery Risk)', fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/notebook_shap_classification_bar.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# SECTION 15 — BUSINESS INSIGHTS
cells.append(md("""## 14. Business Insights <a id="insights"></a>

Insights are structured as: **FACT → INSIGHT → RISK/OPPORTUNITY → ACTION**

---

### Insight 1
**FACT:** 54.8% of all shipments in the dataset are delivered late.  
**INSIGHT:** Late delivery is not an exception — it is the statistical norm in this supply chain. The issue is systemic, not isolated.  
**RISK/OPPORTUNITY:** Customer satisfaction is being consistently compromised, creating both retention risk and brand risk.  
**ACTION:** Establish a supply chain performance improvement programme with measurable KPIs (target: reduce late rate below 40% within 12 months).

---

### Insight 2
**FACT:** Standard Class shipping accounts for ~60% of all orders and has the highest absolute volume of late deliveries.  
**INSIGHT:** The dominant shipping mode is the primary driver of the overall late-delivery statistic.  
**RISK/OPPORTUNITY:** Improving Standard Class performance would have the largest impact on the overall KPI.  
**ACTION:** Audit Standard Class carrier SLA compliance; renegotiate or diversify logistics partners for Standard Class routes.

---

### Insight 3
**FACT:** Scheduled delivery days is the most important predictive factor for actual delivery time across both the regression and classification models (top SHAP feature).  
**INSIGHT:** The scheduling system's own estimates are informative — but the model captures additional signal from region, category, and shipping mode combinations that the scheduling system does not account for.  
**RISK/OPPORTUNITY:** If scheduling systematically underestimates delivery time for specific combinations (e.g., Standard Class + Africa), those misestimates are predictable.  
**ACTION:** Run the model on scheduled orders and flag cases where the model's predicted delivery days exceed the scheduled days — these are high-risk orders.

---

### Insight 4
**FACT:** The Gradient Boosting classifier achieves ROC-AUC = 0.800, indicating meaningful predictive discrimination using only pre-delivery information.  
**INSIGHT:** It is possible to identify high-risk shipments before they are dispatched.  
**RISK/OPPORTUNITY:** Each year of undeployed predictive capability represents preventable late deliveries and associated costs.  
**ACTION:** Integrate the prediction API into the order management system. For HIGH-risk shipments (predicted late probability > 65%), trigger alerts to the dispatch team.

---

### Insight 5
**FACT:** Same Day delivery mode has the lowest late-delivery rate but represents only ~5% of shipments.  
**INSIGHT:** Premium shipping reliably meets delivery commitments.  
**RISK/OPPORTUNITY:** Opportunity to improve customer experience for high-value customers by routing them to more reliable modes.  
**ACTION:** Define eligibility criteria for expedited shipping upgrades and implement an automated upgrade rule for Corporate segment orders above a revenue threshold.
"""))

# SECTION 16 — PREDICTION PIPELINE
cells.append(md("""## 15. Prediction Pipeline <a id="prediction"></a>

The `src/prediction.py` module provides a unified inference interface.
"""))
cells.append(code("""from src.prediction import load_pipeline, predict_single, assign_risk_category

# Load saved artifacts
artifacts = load_pipeline('../models')
print("Pipeline loaded successfully.")
print(f"Regression features: {len(artifacts['regression_features'])}")
print(f"Classification features: {len(artifacts['classification_features'])}")

# Example: predict for a sample order
sample_order = {f: float(X_test.iloc[0][f]) for f in artifacts['regression_features'] if f in X_test.columns}

result = predict_single(sample_order, artifacts)
print(f"\\nExample prediction:")
print(f"  Predicted delivery days: {result['predicted_delivery_days']}")
print(f"  Late probability:        {result['late_probability']:.1%}")
print(f"  Risk category:           {result['risk_category']}")
print(f"  Actual delivery days:    {y_reg_test.iloc[0]}")
print(f"  Actually late:           {'Yes' if y_clf_test.iloc[0] == 1 else 'No'}")"""))

# SECTION 17 — RECOMMENDATIONS
cells.append(md("""## 16. Risk Analysis & Recommendations <a id="recommendations"></a>

### Risk Assessment Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Standard Class capacity bottleneck | High | High | Carrier diversification |
| Regional infrastructure gaps | Medium | High | Route optimisation, local partnerships |
| Scheduling algorithm misestimation | High | Medium | Recalibrate scheduling with model predictions |
| High-value order delays | Low | High | Automatic upgrade to expedited mode |
| Seasonal volume spikes | Medium | Medium | Capacity planning using trend analysis |

### Top Recommendations

1. **Deploy the predictive model in production** — integrate the prediction API at order confirmation to score all new shipments in real-time.
2. **Revise Standard Class SLA** — Standard Class accounts for 60% of volume; improving its performance is the highest-leverage intervention.
3. **Create a risk-tiered intervention protocol:** HIGH risk → dispatch alert + customer notification; MEDIUM risk → customer notification; LOW risk → standard process.
4. **Recalibrate scheduling estimates** — use the regression model's predictions to flag cases where scheduling underestimates delivery time.
5. **Regional performance management** — establish region-specific KPIs and conduct quarterly logistics partner reviews for underperforming regions.
6. **Upgrade high-value Corporate customers** — automatic mode upgrades for orders above a revenue threshold to protect the highest-value relationships.
"""))

# SECTION 18 — CONCLUSION
cells.append(md("""## 17. Conclusion <a id="conclusion"></a>

This project demonstrated that supply chain delivery risk can be meaningfully predicted from pre-delivery information alone, without relying on any post-delivery data leakage.

### Key Technical Achievements
- Built a reproducible end-to-end ML pipeline with strict leakage prevention
- Achieved ROC-AUC = 0.800 for late-delivery risk classification
- Achieved RMSE = 1.259 days for delivery time regression
- Identified scheduled delivery days, shipping mode, and order region as the strongest predictive factors via SHAP analysis
- Deployed an interactive 3-page Streamlit executive dashboard

### Key Business Findings
1. 54.8% of shipments are late — this is a systemic, not isolated, problem
2. Standard Class shipping is the primary driver of late deliveries by volume
3. The predictive model can flag high-risk shipments before dispatch, enabling proactive intervention
4. Same Day delivery reliably meets SLAs but is significantly underutilised
5. Scheduled delivery days is the most informative pre-delivery predictor

### Limitations
- The dataset covers 2015–2018; supply chain patterns may have changed post-COVID
- The regression R² of 0.39 indicates substantial unexplained variance — additional features (carrier performance, weather, inventory availability) could improve the model
- SHAP feature importance shows association, not causation
- Optimal probability thresholds depend on the operational cost of false alarms vs. missed detections, which requires business input

### Future Work
- Real-time data integration with order management systems
- Carrier-level performance modelling
- External feature integration (weather, geopolitical risk indices)
- Time-series models (LSTM, Prophet) for volume forecasting
- Automated retraining pipeline with data drift detection

---

**Author:** Niveditha | IBM SkillsBuild Data Analytics with AI Internship  
**Repository:** https://github.com/[username]/supply-chain-risk-delivery-prediction
"""))

# ============================================================
# Assemble notebook
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.4",
        },
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {NB_PATH}")
print(f"Total cells: {len(cells)}")
