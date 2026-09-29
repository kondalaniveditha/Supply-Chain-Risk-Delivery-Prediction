# ============================================================
# scripts/build_notebook.py
# ============================================================
# Generates the complete Jupyter notebook programmatically.
#
# Run from project root:
#     venv/Scripts/python scripts/build_notebook.py
# ============================================================

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

**Author:** K. Niveditha Sai  
**Programme:** IBM SkillsBuild Data Analytics with AI Internship  
**Dataset:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS (DataCo Global, 180,519 records)  
**Year:** 2026

---

## Abstract

This project develops an end-to-end machine learning system for supply chain delivery analytics. Using 180,519 historical shipment records from the DataCo Global supply chain dataset, the system performs two predictive tasks: (1) regression to estimate actual delivery time in days, and (2) binary classification to estimate the probability of late delivery.

A leakage-prevention policy is applied so that model inputs exclude variables that directly reveal the delivery outcome or become available only after the outcome. In particular, `Delivery Status`, `shipping date (DateOrders)`, `Order Status`, both target variables, and personally identifiable information are excluded from the final modelling inputs.

The best regression model is Random Forest, achieving MAE = 0.954 days, RMSE = 1.259 days, and R² = 0.393 on the chronological test set. The selected classification model is XGBoost based on the project's primary F1-score selection criterion, achieving Accuracy = 0.716, Precision = 0.840, Recall = 0.593, F1 = 0.695, and ROC-AUC = 0.768.

SHAP-based explainability is used to identify features that contribute most strongly to model predictions. These explanations describe predictive associations and should not be interpreted as evidence of causation.

The project also includes an interactive Streamlit dashboard for delivery analytics, risk prediction, model interpretation, and operational decision support.
 
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


# ============================================================
# SECTION 2 — PROBLEM STATEMENT
# ============================================================

cells.append(md("""## 1. Problem Statement & Objectives <a id="problem"></a>

### Problem Statement

Supply chain delivery performance is an important operational consideration because delays can affect service levels, logistics planning, inventory coordination, and customer experience. A reactive approach identifies delays only after they occur. A predictive approach can instead provide an early risk signal using information available around the time an order is placed.

The goal of this project is therefore to build a predictive analytics system that:

- estimates expected delivery time;
- estimates the probability of late delivery;
- identifies important predictive features;
- provides interpretable model outputs; and
- supports proactive analysis of potentially higher-risk shipments.

The model outputs are intended as **decision-support signals**, not as guaranteed delivery outcomes.

### Data Leakage Prevention Policy

> **CRITICAL:** This project is designed as a pre-delivery prediction system. The following columns are excluded from the final model inputs because they either directly represent the outcome or may contain information that becomes available after the delivery process:

> - `Delivery Status` — directly describes delivery outcome/status
> - `shipping date (DateOrders)` — represents shipment timing that may not be available at order placement
> - `Order Status` — post-order lifecycle information that can encode outcome-related information
> - `Late_delivery_risk` — classification target
> - `Days for shipping (real)` — regression target
> - All PII columns

The purpose of this exclusion is to prevent the models from learning directly from information that would not be appropriate for a pre-delivery prediction scenario.

### Objectives

1. Audit and understand the DataCo supply chain dataset.
2. Implement a reproducible preprocessing pipeline with leakage prevention.
3. Conduct exploratory data analysis to identify patterns in delivery performance.
4. Build and compare regression models for delivery time prediction.
5. Build and compare classification models for late-delivery risk prediction.
6. Apply SHAP-based explainability to interpret model predictions.
7. Translate observed patterns and model outputs into practical business recommendations.
8. Deploy an interactive executive-style dashboard.
"""))


# ============================================================
# SECTION 3 — BUSINESS CONTEXT
# ============================================================

cells.append(md("""## 2. Business Context <a id="business"></a>

Supply chain analytics can support several operational areas, including:

- **Delivery Performance** — understanding actual and scheduled delivery patterns.
- **Operational Planning** — identifying shipment segments with higher observed delay rates.
- **Risk Management** — estimating the probability of late delivery before the final outcome is known.
- **Capacity Planning** — examining shipment volume and temporal patterns.
- **Service Management** — comparing delivery performance across shipping modes, markets, and regions.

The DataCo dataset covers shipments across five global markets:

- Africa
- Europe
- LATAM
- Pacific Asia
- USCA

The dataset contains historical records from 2015–2018.

### Business Question

> **Given information available before the delivery outcome is known, what is the expected delivery time and what is the estimated probability that the shipment will be late?**

The answer is treated as a predictive estimate rather than a guaranteed outcome.
"""))


# ============================================================
# SECTION 4 — SETUP
# ============================================================

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
    load_raw_data,
    report_data_quality,
    parse_dates,
    create_datetime_features,
    drop_leakage_and_pii,
    handle_missing_values,
    encode_categoricals,
    build_feature_matrix,
    chronological_split,
    run_full_pipeline,
    LEAKAGE_COLS,
    PII_COLS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

from src.feature_engineering import (
    add_all_features,
    get_final_feature_columns
)

from src.regression_models import (
    train_regression_models,
    save_regression_artifacts
)

from src.classification_models import (
    train_classification_models,
    save_classification_artifacts
)

from src.evaluation import (
    compare_regression_models,
    compare_classification_models,
    select_best_regression_model,
    select_best_classification_model,
    get_confusion_matrix
)

from src.explainability import (
    compute_shap_values,
    get_top_shap_features
)

sns.set_theme(style='whitegrid', palette='muted')

FIGURES_DIR = '../reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

print('Environment ready.')
"""))


# ============================================================
# SECTION 5 — DATASET OVERVIEW
# ============================================================

cells.append(md("""## 4. Dataset Overview <a id="dataset"></a>

**Source:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS  
**Reference:** Constante, F., Silva, F., & Pereira, A. (2019). DataCo Smart Supply Chain for Big Data Analysis. Mendeley Data.  
**File:** `data/raw/DataCoSupplyChainDataset.csv`  
**Encoding:** Latin-1

The dataset contains 180,519 shipment/order records and 53 original columns. It includes information related to orders, customers, products, shipping, markets, regions, financial attributes, and delivery outcomes.
"""))

cells.append(code("""df_raw = load_raw_data(
    '../data/raw/DataCoSupplyChainDataset.csv'
)

print(
    f"Dataset shape: "
    f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns"
)

print("\\nColumn names:")

for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col}")
"""))

cells.append(code("""print("Basic statistics for key numerical variables:")

key_cols = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Sales',
    'Order Item Quantity',
    'Product Price'
]

df_raw[key_cols].describe().round(3)
"""))

cells.append(code("""print("Target variable distributions:")

print("\\nDays for shipping (real):")
print(
    df_raw['Days for shipping (real)']
    .value_counts()
    .sort_index()
)

print("\\nLate_delivery_risk:")
print(
    df_raw['Late_delivery_risk']
    .value_counts()
)

late_pct = df_raw['Late_delivery_risk'].mean() * 100

print(
    f"\\nLate delivery rate: {late_pct:.1f}%"
)
"""))


# ============================================================
# SECTION 6 — DATA QUALITY
# ============================================================

cells.append(md("""## 5. Data Quality Analysis <a id="quality"></a>

A systematic quality audit is performed before modelling. The audit examines:

- missing values;
- data types;
- duplicate records;
- numerical distributions; and
- potentially problematic fields.

This step helps identify data-quality issues before preprocessing and model development.
"""))

cells.append(code("""quality = report_data_quality(df_raw)

print("Data Quality Report:")
print(quality.to_string())
"""))

cells.append(code("""print("Missing value summary:")

missing = quality[
    quality['missing_count'] > 0
][
    ['dtype', 'missing_count', 'missing_pct']
]

print(missing)

print(
    f"\\nDuplicate rows: {df_raw.duplicated().sum()}"
)
"""))


# ============================================================
# SECTION 7 — DATA CLEANING
# ============================================================

cells.append(md("""## 6. Data Cleaning & Preprocessing <a id="cleaning"></a>

The preprocessing pipeline performs the following major operations:

1. **Load raw CSV** — read the DataCo dataset using Latin-1 encoding.
2. **Parse order dates** — convert date fields into datetime representations.
3. **Create datetime features** — extract year, month, quarter, day, day-of-week, and hour where applicable.
4. **Remove leakage and PII** — exclude target variables, outcome-related information, identifiers, and personally identifiable information.
5. **Create the chronological train/test split** — earlier orders are used for training and later orders are used for testing.
6. **Handle missing values** — numerical and categorical missing values are processed using the project's preprocessing functions.
7. **Encode categorical variables** — categorical predictors are transformed into numerical representations.
8. **Build feature matrices** — separate matrices are created for regression and classification.
9. **Save processed data** — processed data snapshots are generated for downstream analysis and dashboard use.

### Chronological Evaluation

A chronological split is used so that the test set represents later records than the training set. This provides a more realistic evaluation setting than randomly mixing historical and later shipments.

> **Important implementation note:** The project uses the preprocessing functions already defined in `src/data_preprocessing.py`. The modelling results reported in this notebook correspond to that implemented pipeline and should be interpreted together with the documented preprocessing assumptions.

> All transformations operate on working copies of the data. The original raw CSV is not overwritten.
"""))

cells.append(code("""print("Running full preprocessing pipeline...")

pipeline_result = run_full_pipeline(
    raw_path='../data/raw/DataCoSupplyChainDataset.csv',
    processed_path='../data/processed/supply_chain_processed.csv'
)

df_full = pipeline_result['df_full']
df_train = pipeline_result['df_train']
df_test = pipeline_result['df_test']
encoders = pipeline_result['encoders']

print(f"\\nPreprocessed full dataset: {len(df_full):,} rows")
print(f"Train split: {len(df_train):,} rows")
print(f"Test  split: {len(df_test):,} rows")

print(
    f"\\nRemaining columns "
    f"({len(df_full.columns)}):"
)

print(df_full.columns.tolist())
"""))

cells.append(code("""# Verify no leakage columns remain

print("Leakage column check:")

for col in LEAKAGE_COLS:
    status = (
        '✗ PRESENT (ERROR)'
        if col in df_full.columns
        else '✓ Removed'
    )

    print(f"  {col}: {status}")


print("\\nPII column check:")

for col in PII_COLS:
    status = (
        '✗ PRESENT (ERROR)'
        if col in df_full.columns
        else '✓ Removed'
    )

    print(f"  {col}: {status}")
"""))


# ============================================================
# SECTION 8 — EDA
# ============================================================

cells.append(md("""## 7. Exploratory Data Analysis <a id="eda"></a>

### Key EDA Questions

1. What is the distribution of actual delivery times?
2. What percentage of shipments are late?
3. How does delivery performance differ across shipping modes?
4. Which markets/regions show different observed delay rates?
5. Are there temporal patterns in delivery performance?
6. Which numerical variables show relationships with delivery time?

EDA findings are treated as descriptive patterns in the historical dataset. They are not interpreted as causal relationships.
"""))

cells.append(code("""# Q1: Delivery time distribution

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

dist = (
    df_raw['Days for shipping (real)']
    .value_counts()
    .sort_index()
)

colors = [
    '#ef4444' if i >= 4 else '#3b82d4'
    for i in dist.index
]

axes[0].bar(
    dist.index,
    dist.values,
    color=colors,
    edgecolor='white'
)

axes[0].set_xlabel(
    'Actual Delivery Days',
    fontsize=12
)

axes[0].set_ylabel(
    'Number of Shipments',
    fontsize=12
)

axes[0].set_title(
    'Distribution of Actual Delivery Days',
    fontsize=13,
    fontweight='bold'
)

axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f'{int(x):,}'
    )
)


# Scheduled vs actual
sched_dist = (
    df_raw['Days for shipment (scheduled)']
    .value_counts()
    .sort_index()
)

axes[1].bar(
    sched_dist.index,
    sched_dist.values,
    color='#7c5cd8',
    edgecolor='white'
)

axes[1].set_xlabel(
    'Scheduled Delivery Days',
    fontsize=12
)

axes[1].set_ylabel(
    'Number of Shipments',
    fontsize=12
)

axes[1].set_title(
    'Distribution of Scheduled Delivery Days',
    fontsize=13,
    fontweight='bold'
)

plt.suptitle(
    'Actual vs Scheduled Delivery Days Distribution',
    fontsize=14,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/eda_delivery_distributions.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print(
    f"Mean actual days: "
    f"{df_raw['Days for shipping (real)'].mean():.2f}"
)

print(
    f"Median actual days: "
    f"{df_raw['Days for shipping (real)'].median():.0f}"
)

print(
    f"Mean scheduled days: "
    f"{df_raw['Days for shipment (scheduled)'].mean():.2f}"
)
"""))

cells.append(code("""# Q2 & Q3: Late delivery by shipping mode

ship_perf = (
    df_raw
    .groupby('Shipping Mode')
    .agg(
        total=(
            'Days for shipping (real)',
            'count'
        ),
        avg_days=(
            'Days for shipping (real)',
            'mean'
        ),
        late_rate=(
            'Late_delivery_risk',
            'mean'
        )
    )
    .reset_index()
    .sort_values(
        'late_rate',
        ascending=False
    )
)

print(
    "Delivery performance by shipping mode:"
)

print(
    ship_perf.to_string(index=False)
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

axes[0].barh(
    ship_perf
    .sort_values('avg_days')['Shipping Mode'],
    ship_perf
    .sort_values('avg_days')['avg_days'],
    color='#3b82d4'
)

axes[0].set_xlabel(
    'Average Delivery Days',
    fontsize=12
)

axes[0].set_title(
    'Avg Delivery Days by Shipping Mode',
    fontsize=12,
    fontweight='bold'
)


axes[1].barh(
    ship_perf
    .sort_values('late_rate')['Shipping Mode'],
    ship_perf
    .sort_values('late_rate')['late_rate'] * 100,
    color='#f59e0b'
)

axes[1].set_xlabel(
    'Late Delivery Rate (%)',
    fontsize=12
)

axes[1].set_title(
    'Late Rate by Shipping Mode',
    fontsize=12,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/eda_shipping_mode_perf.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))

cells.append(code("""# Q4: Market and regional performance

mkt_perf = (
    df_raw
    .groupby('Market')
    .agg(
        avg_days=(
            'Days for shipping (real)',
            'mean'
        ),
        late_rate=(
            'Late_delivery_risk',
            'mean'
        ),
        count=(
            'Days for shipping (real)',
            'count'
        )
    )
    .reset_index()
    .sort_values(
        'late_rate',
        ascending=False
    )
)

print("Delivery performance by market:")

print(
    mkt_perf.to_string(index=False)
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

axes[0].barh(
    mkt_perf
    .sort_values('late_rate')['Market'],
    mkt_perf
    .sort_values('late_rate')['late_rate'] * 100,
    color=[
        '#ef4444' if r > 0.55
        else '#3b82d4'
        for r in
        mkt_perf
        .sort_values('late_rate')['late_rate']
    ]
)

axes[0].set_xlabel(
    'Late Delivery Rate (%)',
    fontsize=12
)

axes[0].set_title(
    'Late Rate by Market',
    fontsize=12,
    fontweight='bold'
)


axes[1].barh(
    mkt_perf
    .sort_values('avg_days')['Market'],
    mkt_perf
    .sort_values('avg_days')['avg_days'],
    color='#7c5cd8'
)

axes[1].set_xlabel(
    'Average Delivery Days',
    fontsize=12
)

axes[1].set_title(
    'Avg Delivery Days by Market',
    fontsize=12,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/eda_market_perf.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))

cells.append(code("""# Q5: Temporal trend

df_raw['order_date'] = pd.to_datetime(
    df_raw['order date (DateOrders)'],
    format='%m/%d/%Y %H:%M',
    errors='coerce'
)

monthly = (
    df_raw
    .set_index('order_date')
    .resample('ME')
    .agg(
        avg_days=(
            'Days for shipping (real)',
            'mean'
        ),
        late_rate=(
            'Late_delivery_risk',
            'mean'
        ),
        volume=(
            'Days for shipping (real)',
            'count'
        )
    )
)


fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 9),
    sharex=True
)

axes[0].plot(
    monthly.index,
    monthly['avg_days'],
    color='#3b82d4',
    linewidth=2
)

axes[0].set_ylabel(
    'Avg Delivery Days',
    fontsize=11
)

axes[0].set_title(
    'Monthly Supply Chain Performance',
    fontsize=13,
    fontweight='bold'
)

axes[0].grid(
    linestyle='--',
    alpha=0.4
)


axes[1].plot(
    monthly.index,
    monthly['late_rate'] * 100,
    color='#ef4444',
    linewidth=2
)

axes[1].axhline(
    df_raw['Late_delivery_risk'].mean() * 100,
    color='black',
    linestyle='--',
    linewidth=1,
    label=(
        f"Avg "
        f"{df_raw['Late_delivery_risk'].mean() * 100:.1f}%"
    )
)

axes[1].set_ylabel(
    'Late Rate (%)',
    fontsize=11
)

axes[1].legend(
    fontsize=9
)

axes[1].grid(
    linestyle='--',
    alpha=0.4
)


axes[2].bar(
    monthly.index,
    monthly['volume'],
    color='#7c5cd8',
    width=25,
    alpha=0.7
)

axes[2].set_ylabel(
    'Shipment Volume',
    fontsize=11
)

axes[2].set_xlabel(
    'Month',
    fontsize=11
)

axes[2].grid(
    linestyle='--',
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/eda_monthly_trend.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))

cells.append(code("""# Q6: Correlation of numerical features with targets

num_corr_cols = (
    NUMERICAL_FEATURES
    + [
        'Days for shipping (real)',
        'Late_delivery_risk'
    ]
)

corr_matrix = df_raw[
    num_corr_cols
].corr()


fig, ax = plt.subplots(
    figsize=(13, 10)
)

import numpy as np

mask = np.triu(
    np.ones_like(
        corr_matrix,
        dtype=bool
    )
)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    ax=ax,
    annot_kws={'size': 8},
    linewidths=0.3,
    vmin=-1,
    vmax=1
)

ax.set_title(
    'Pearson Correlation Heatmap — Numerical Features',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/eda_correlation_heatmap.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()


print(
    "\\nCorrelation with Days for shipping (real):"
)

corr_target = (
    corr_matrix[
        'Days for shipping (real)'
    ]
    .drop('Days for shipping (real)')
    .sort_values()
)

print(
    corr_target.to_string()
)
"""))


# ============================================================
# SECTION 9 — FEATURE ENGINEERING
# ============================================================

cells.append(md("""## 8. Feature Engineering <a id="features"></a>

Beyond the basic preprocessing, additional features are created using information available from the order and shipment context.

| Feature | Description | Source |
|---------|-------------|--------|
| `order_year/month/quarter/day/dayofweek/hour` | Date components | Order date |
| `is_expedited_schedule` | 1 if scheduled days ≤ 2 | Scheduled days |
| `profit_margin_ratio` | Order Profit / Order Item Total | Order financials |
| `has_discount` | 1 if item discount > 0 | Item discount |
| `is_high_value` | 1 if Product Price ≥ $200 | Product price |
| `is_weekend_order` | 1 if order placed on Sat/Sun | Order day-of-week |

These engineered variables are intended to provide the models with additional structured information while avoiding direct use of the delivery outcome.
"""))

cells.append(code("""# Apply feature engineering to train/test splits

X_train_raw = None
X_test_raw = None

y_reg_train = None
y_reg_test = None

y_clf_train = None
y_clf_test = None


df_train_fe = add_all_features(
    df_train
)

df_test_fe = add_all_features(
    df_test
)


X_train, y_reg_train, y_clf_train = (
    build_feature_matrix(
        df_train_fe
    )
)

X_test, y_reg_test, y_clf_test = (
    build_feature_matrix(
        df_test_fe
    )
)


# Keep only numeric columns
num_cols = (
    X_train
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

X_train = X_train[
    num_cols
]

X_test = X_test.reindex(
    columns=num_cols,
    fill_value=0
)


print(
    f"Final feature set "
    f"({len(num_cols)} features):"
)

for i, col in enumerate(
    num_cols,
    1
):
    print(
        f"  {i:2d}. {col}"
    )


print(
    f"\\nTrain: {X_train.shape}, "
    f"Test: {X_test.shape}"
)

print(
    f"Late rate (train): "
    f"{y_clf_train.mean() * 100:.1f}%"
)

print(
    f"Late rate (test):  "
    f"{y_clf_test.mean() * 100:.1f}%"
)
"""))


# ============================================================
# SECTION 10 — MACHINE LEARNING METHODOLOGY
# ============================================================

cells.append(md("""## 9. Machine Learning Methodology <a id="methodology"></a>

### Train/Test Split Strategy

A **chronological split** is used: the earliest 80% of orders form the training set and the latest 20% form the test set.

This evaluation design is intended to reflect a practical situation in which historical shipments are used to train models and later shipments are used to evaluate their predictive performance.

### Models Evaluated

**Regression — predicting delivery days:**

- Baseline Mean Predictor
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

**Classification — predicting late-delivery risk:**

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Class Distribution

The classification target contains approximately 55% late and 45% on-time shipments. This represents a moderate class imbalance rather than an extreme imbalance.

`class_weight='balanced'` is used in applicable models to reduce the influence of class-frequency differences during training.

### Feature Scaling

`StandardScaler` is applied to the feature matrices used by the modelling pipeline. The project evaluates models using the transformed numerical representation.
"""))


# ============================================================
# SECTION 11 — REGRESSION
# ============================================================

cells.append(md("""## 10. Delivery Time Prediction (Regression) <a id="regression"></a>

**Target:** `Days for shipping (real)`

The regression task estimates the actual number of days required to deliver a shipment.
"""))

cells.append(code("""from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print(
    "Training regression models..."
)

reg_models, reg_scaler = (
    train_regression_models(
        X_train,
        y_reg_train
    )
)

X_test_reg = (
    reg_scaler.transform(
        X_test
    )
)

reg_comparison = (
    compare_regression_models(
        reg_models,
        X_test_reg,
        y_reg_test
    )
)

print(
    "\\nRegression Model Comparison:"
)

print(
    reg_comparison.to_string(
        index=False
    )
)
"""))

cells.append(code("""# Load pre-computed comparison if available

import json

if os.path.exists(
    '../models/training_summary.json'
):

    with open(
        '../models/training_summary.json'
    ) as f:

        summary = json.load(f)

    reg_best = (
        summary[
            'regression'
        ][
            'best_model'
        ]
    )

    print(
        f"Best Regression Model: "
        f"{reg_best['model']}"
    )

    print(
        f"  MAE  = "
        f"{reg_best['MAE']}"
    )

    print(
        f"  RMSE = "
        f"{reg_best['RMSE']}"
    )

    print(
        f"  R²   = "
        f"{reg_best['R2']}"
    )
"""))

cells.append(code("""best_reg_name, best_reg_model = (
    select_best_regression_model(
        reg_comparison,
        reg_models
    )
)

save_regression_artifacts(
    best_reg_model,
    reg_scaler,
    num_cols,
    '../models'
)

print(
    f"Best model saved: "
    f"{best_reg_name}"
)


# Visualise actual vs predicted

y_reg_pred = (
    best_reg_model.predict(
        X_test_reg
    )
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


axes[0].scatter(
    y_reg_test,
    y_reg_pred,
    alpha=0.1,
    s=8,
    color='#3b82d4'
)

mn = min(
    y_reg_test.min(),
    y_reg_pred.min()
)

mx = max(
    y_reg_test.max(),
    y_reg_pred.max()
)

axes[0].plot(
    [mn, mx],
    [mn, mx],
    'r--',
    linewidth=1.5,
    label='Perfect prediction'
)

axes[0].set_xlabel(
    'Actual Delivery Days',
    fontsize=12
)

axes[0].set_ylabel(
    'Predicted Delivery Days',
    fontsize=12
)

axes[0].set_title(
    f'{best_reg_name} — Actual vs Predicted',
    fontsize=12,
    fontweight='bold'
)

axes[0].legend(
    fontsize=10
)


residuals = (
    y_reg_test
    - y_reg_pred
)

axes[1].hist(
    residuals,
    bins=30,
    color='#7c5cd8',
    edgecolor='white'
)

axes[1].axvline(
    0,
    color='red',
    linestyle='--',
    linewidth=1.5
)

axes[1].set_xlabel(
    'Residual (Actual − Predicted)',
    fontsize=12
)

axes[1].set_ylabel(
    'Count',
    fontsize=12
)

axes[1].set_title(
    'Residual Distribution',
    fontsize=12,
    fontweight='bold'
)

plt.suptitle(
    'Regression Model Diagnostics',
    fontsize=14,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/regression_diagnostics.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print(
    f"Residual mean: "
    f"{residuals.mean():.4f}, "
    f"std: {residuals.std():.4f}"
)
"""))


# ============================================================
# SECTION 12 — CLASSIFICATION
# ============================================================

cells.append(md("""## 11. Delivery Risk Prediction (Classification) <a id="classification"></a>

**Target:** `Late_delivery_risk`

The classification task estimates whether a shipment belongs to the late-delivery class.

- `0` = on time
- `1` = late

The model produces both a class prediction and, where supported, a probability estimate.
"""))

cells.append(code("""print(
    "Training classification models..."
)

clf_models, clf_scaler = (
    train_classification_models(
        X_train,
        y_clf_train
    )
)

X_test_clf = (
    clf_scaler.transform(
        X_test
    )
)

clf_comparison = (
    compare_classification_models(
        clf_models,
        X_test_clf,
        y_clf_test
    )
)

print(
    "\\nClassification Model Comparison:"
)

print(
    clf_comparison.to_string(
        index=False
    )
)
"""))

cells.append(code("""best_clf_name, best_clf_model = (
    select_best_classification_model(
        clf_comparison,
        clf_models
    )
)

save_classification_artifacts(
    best_clf_model,
    clf_scaler,
    num_cols,
    '../models'
)

print(
    f"Best model saved: "
    f"{best_clf_name}"
)
"""))

cells.append(code("""from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import numpy as np


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


# Confusion matrix

y_clf_pred = (
    best_clf_model.predict(
        X_test_clf
    )
)

cm = get_confusion_matrix(
    y_clf_test,
    y_clf_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        'On Time',
        'Late'
    ]
)

disp.plot(
    ax=axes[0],
    colorbar=False,
    cmap='Blues'
)

axes[0].set_title(
    f'Confusion Matrix — {best_clf_name}',
    fontsize=12,
    fontweight='bold'
)


# ROC curves

for name, model in clf_models.items():

    if hasattr(
        model,
        'predict_proba'
    ):

        y_prob = (
            model
            .predict_proba(
                X_test_clf
            )[:, 1]
        )

        fpr, tpr, _ = (
            roc_curve(
                y_clf_test,
                y_prob
            )
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        axes[1].plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f'{name} '
                f'(AUC={roc_auc:.3f})'
            )
        )

axes[1].plot(
    [0, 1],
    [0, 1],
    'k--',
    linewidth=1
)

axes[1].set_xlabel(
    'False Positive Rate',
    fontsize=12
)

axes[1].set_ylabel(
    'True Positive Rate',
    fontsize=12
)

axes[1].set_title(
    'ROC Curves — All Classifiers',
    fontsize=12,
    fontweight='bold'
)

axes[1].legend(
    fontsize=9
)

axes[1].grid(
    linestyle='--',
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/classification_roc_confusion.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))


# ============================================================
# SECTION 13 — MODEL COMPARISON
# ============================================================

cells.append(md("""## 12. Model Evaluation & Comparison <a id="evaluation"></a>

### Regression Results

The following results were obtained on the chronological held-out test set.

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Random Forest | 0.9539 | 1.2590 | 0.3929 |
| Gradient Boosting | 0.9682 | 1.2618 | 0.3902 |
| XGBoost | 0.9745 | 1.2634 | 0.3887 |
| Linear Regression | 0.9840 | 1.2643 | 0.3878 |
| Baseline (Mean) | 1.4186 | 1.6158 | 0.0000 |

**Selected regression model: Random Forest**, based on the project's lowest RMSE selection criterion.

### Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| XGBoost | 0.7160 | 0.8404 | 0.5930 | 0.6953 | 0.7684 |
| Gradient Boosting | 0.7161 | 0.8407 | 0.5928 | 0.6953 | 0.7669 |
| Random Forest | 0.7178 | 0.8500 | 0.5873 | 0.6947 | 0.7685 |
| Logistic Regression | 0.6936 | 0.8424 | 0.5404 | 0.6584 | 0.7394 |

**Selected classification model: XGBoost**, based on the project's primary F1-score selection criterion.

XGBoost and Gradient Boosting both achieve an F1 score of 0.6953 at the displayed precision. XGBoost is retained as the selected model according to the project's selection procedure.

Random Forest has a marginally higher ROC-AUC of 0.7685 compared with XGBoost's 0.7684. Therefore, XGBoost should not be described as having the highest ROC-AUC.

### Interpretation Notes

- The regression R² of approximately 0.39 indicates that the selected regression model explains about 39% of the observed variance in delivery time on the held-out test set.
- The remaining unexplained variance indicates that additional factors not available in the current feature set may contribute to delivery-time variation.
- The classification ROC-AUC of approximately 0.768 indicates useful discrimination between late and on-time shipments in this test set.
- Precision of approximately 0.84 with recall of approximately 0.59 means that the classifier identifies a subset of late shipments with relatively high precision, while a substantial portion of actual late shipments are not captured at the default classification threshold.
- These results should be interpreted as historical test-set performance rather than a guarantee of future operational performance.
"""))


# ============================================================
# SECTION 14 — XAI
# ============================================================

cells.append(md("""## 13. Explainable AI (SHAP) <a id="xai"></a>

SHAP (SHapley Additive exPlanations) values quantify how strongly individual features contribute to model predictions.

The analysis below provides:

- global feature importance;
- relative contribution strength;
- model-level interpretation; and
- a basis for investigating individual predictions.

> **Important:** SHAP values describe how features are associated with model outputs. They do **not** establish causation.

Therefore, statements such as **"important predictive feature"** or **"associated with model predictions"** are preferred over statements such as **"causes late delivery."**
"""))

cells.append(code("""import shap
import os

shap.initjs()


X_train_reg_scaled = (
    reg_scaler.transform(
        X_train
    )
)

X_test_reg_scaled = (
    reg_scaler.transform(
        X_test
    )
)

X_train_clf_scaled = (
    clf_scaler.transform(
        X_train
    )
)

X_test_clf_scaled = (
    clf_scaler.transform(
        X_test
    )
)


# SHAP for best regression model

print(
    "Computing SHAP values for regression model..."
)

shap_vals_reg, X_sample_reg = (
    compute_shap_values(
        best_reg_model,
        X_test_reg_scaled,
        X_train_reg_scaled,
        sample_size=1000
    )
)

top_reg = (
    get_top_shap_features(
        shap_vals_reg,
        num_cols,
        top_n=10
    )
)

print(
    "\\nTop 10 features "
    "(regression - by mean |SHAP|):"
)

print(
    top_reg.to_string(
        index=False
    )
)
"""))

cells.append(code("""# Plot SHAP bar for regression

import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(
    figsize=(9, 6)
)

mean_abs = (
    np.abs(
        shap_vals_reg
    ).mean(axis=0)
)

imp_df = (
    top_reg
    .sort_values(
        'mean_abs_shap'
    )
)

ax.barh(
    imp_df['feature'],
    imp_df['mean_abs_shap'],
    color='#3b82d4'
)

ax.set_xlabel(
    'Mean |SHAP value|',
    fontsize=12
)

ax.set_title(
    f'SHAP Global Importance — '
    f'{best_reg_name} (Delivery Days)',
    fontsize=13,
    fontweight='bold'
)

ax.grid(
    axis='x',
    linestyle='--',
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/notebook_shap_regression_bar.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))

cells.append(code("""# SHAP for best classification model

print(
    "Computing SHAP values for classification model..."
)

shap_vals_clf, X_sample_clf = (
    compute_shap_values(
        best_clf_model,
        X_test_clf_scaled,
        X_train_clf_scaled,
        sample_size=1000
    )
)

top_clf = (
    get_top_shap_features(
        shap_vals_clf,
        num_cols,
        top_n=10
    )
)

print(
    "\\nTop 10 features "
    "(classification - by mean |SHAP|):"
)

print(
    top_clf.to_string(
        index=False
    )
)
"""))

cells.append(code("""fig, ax = plt.subplots(
    figsize=(9, 6)
)

imp_clf = (
    top_clf
    .sort_values(
        'mean_abs_shap'
    )
)

ax.barh(
    imp_clf['feature'],
    imp_clf['mean_abs_shap'],
    color='#ef4444'
)

ax.set_xlabel(
    'Mean |SHAP value|',
    fontsize=12
)

ax.set_title(
    f'SHAP Global Importance — '
    f'{best_clf_name} (Late Delivery Risk)',
    fontsize=13,
    fontweight='bold'
)

ax.grid(
    axis='x',
    linestyle='--',
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    f'{FIGURES_DIR}/notebook_shap_classification_bar.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()
"""))


# ============================================================
# SECTION 15 — BUSINESS INSIGHTS
# ============================================================

cells.append(md("""## 14. Business Insights <a id="insights"></a>

Insights are structured as:

**FACT → INSIGHT → RISK/OPPORTUNITY → ACTION**

The following statements are based on the observed dataset and model results. They should be interpreted as descriptive or predictive findings rather than causal conclusions.
"""))

cells.append(md("""### Insight 1

**FACT:** Approximately 54.8% of shipments in the dataset are classified as late.

**INSIGHT:** More than half of the historical shipments in this dataset fall into the late-delivery class.

**RISK/OPPORTUNITY:** A high observed late-delivery rate indicates that delivery performance is an important area for operational analysis. Customer satisfaction and retention are not directly measured in the dataset, so their effects cannot be quantified from these records alone.

**ACTION:** Track late-delivery rate as a core KPI and investigate the shipment characteristics associated with higher observed delay rates.
"""))

cells.append(md("""### Insight 2

**FACT:** Standard Class shipping accounts for approximately 60% of orders and has the highest absolute volume of late deliveries.

**INSIGHT:** Because Standard Class represents a large share of shipments, it also contributes the largest absolute number of late deliveries in this dataset.

**RISK/OPPORTUNITY:** Reviewing Standard Class performance may address a substantial portion of the observed late-delivery volume.

**ACTION:** Compare Standard Class performance across markets, regions, order types, and other available shipment attributes before considering carrier or SLA changes.
"""))

cells.append(md("""### Insight 3

**FACT:** Scheduled delivery days is among the strongest predictive factors identified by the model analysis.

**INSIGHT:** The model uses scheduling information together with shipping, order, regional, and other available features to distinguish delivery outcomes.

**RISK/OPPORTUNITY:** Differences between scheduled and predicted delivery time can be investigated for shipment segments with higher predicted risk.

**ACTION:** Use model predictions as a decision-support signal and review cases where predicted delivery days exceed the scheduled delivery window.
"""))

cells.append(md("""### Insight 4

**FACT:** The selected classification model, XGBoost, achieves ROC-AUC ≈ 0.768 on the held-out chronological test set.

**INSIGHT:** The available pre-delivery features contain useful predictive signal for distinguishing late and on-time shipments.

**RISK/OPPORTUNITY:** A predictive score could support prioritisation of shipments for review, but its operational value should be validated using real intervention outcomes.

**ACTION:** Pilot the prediction system in a controlled workflow and monitor precision, recall, false alerts, missed late deliveries, and business outcomes before production deployment.
"""))

cells.append(md("""### Insight 5

**FACT:** Same Day delivery has the lowest observed late-delivery rate among the shipping modes in this dataset, while representing approximately 5% of shipments.

**INSIGHT:** Same Day shipping shows a lower observed late-delivery rate than the other shipping modes in this historical dataset.

**RISK/OPPORTUNITY:** This pattern can be investigated alongside shipping cost, customer segment, order value, and service requirements.

**ACTION:** Evaluate expedited-shipping options using predicted delivery risk, service requirements, cost, and operational constraints rather than assuming that shipping mode alone guarantees delivery performance.
"""))


# ============================================================
# SECTION 16 — PREDICTION PIPELINE
# ============================================================

cells.append(md("""## 15. Prediction Pipeline <a id="prediction"></a>

The `src/prediction.py` module provides a unified inference interface for the saved regression and classification models.

### Important Input Requirement

The current saved inference artifacts expect the **numeric, preprocessed feature representation used during model training**.

They do not contain the full categorical preprocessing pipeline required to convert an arbitrary raw order record into the trained feature representation.

Therefore:

1. raw order data should first pass through the project's preprocessing and feature-engineering pipeline;
2. the resulting numeric feature values should match the saved model's feature names;
3. the transformed feature dictionary can then be passed to `predict_single()`.

The example below intentionally uses a preprocessed test shipment rather than pretending that raw categorical order data can be passed directly into the saved scaler/model artifacts.
"""))

cells.append(code("""from src.prediction import (
    load_pipeline,
    predict_single,
    assign_risk_category
)


# Load saved artifacts

artifacts = load_pipeline(
    '../models'
)

print(
    "Pipeline loaded successfully."
)

print(
    f"Regression features: "
    f"{len(artifacts['regression_features'])}"
)

print(
    f"Classification features: "
    f"{len(artifacts['classification_features'])}"
)


# Example: predict for a preprocessed test shipment

sample_order = {
    f: float(
        X_test.iloc[0][f]
    )
    for f in artifacts[
        'regression_features'
    ]
    if f in X_test.columns
}


result = predict_single(
    sample_order,
    artifacts
)

print("\\nExample prediction:")

print(
    f"  Predicted delivery days: "
    f"{result['predicted_delivery_days']}"
)

print(
    f"  Late probability:        "
    f"{result['late_probability']:.1%}"
)

print(
    f"  Risk category:           "
    f"{result['risk_category']}"
)

print(
    f"  Actual delivery days:    "
    f"{y_reg_test.iloc[0]}"
)

print(
    f"  Actually late:           "
    f"{'Yes' if y_clf_test.iloc[0] == 1 else 'No'}"
)
"""))


# ============================================================
# SECTION 17 — RISK ANALYSIS & RECOMMENDATIONS
# ============================================================

cells.append(md("""## 16. Risk Analysis & Recommendations <a id="recommendations"></a>

### Risk Assessment Matrix

> **Threshold note:** The current dashboard/inference layer uses fixed probability bands for LOW, MEDIUM, and HIGH risk. These thresholds are **illustrative starting points**, not operationally calibrated decision thresholds. Production use should calibrate them against the cost of false positives, missed late deliveries, and the organisation's intervention capacity.

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| High shipment volume in Standard Class | High | High | Segment-level performance analysis and capacity review |
| Regional performance differences | Medium | High | Region-specific monitoring and investigation |
| Scheduling discrepancy | High | Medium | Compare scheduled time with model predictions |
| High-value order delays | Low | High | Evaluate targeted intervention using risk and cost |
| Seasonal volume changes | Medium | Medium | Capacity planning using historical trends |

### Top Recommendations

1. **Pilot the predictive model in a controlled workflow** — use the prediction API as decision support and monitor prediction quality before considering production deployment.

2. **Review Standard Class performance** — Standard Class represents a large share of shipment volume. Compare its observed performance across markets, regions, and other shipment segments before considering SLA or carrier changes.

3. **Create a risk-tiered intervention protocol** — define HIGH, MEDIUM, and LOW actions after validating probability thresholds against false-alert costs, missed late deliveries, and available intervention capacity.

4. **Review scheduling estimates** — use regression predictions to flag cases where predicted delivery days exceed the scheduled delivery window and investigate the underlying shipment segments.

5. **Regional performance management** — establish region-specific KPIs and investigate consistently higher observed late-delivery rates.

6. **Evaluate expedited-shipping options** — assess potential upgrades using predicted risk, service requirements, shipping cost, customer segment, and operational constraints.

7. **Monitor model performance over time** — because the source dataset covers 2015–2018, future deployment would require monitoring for changes in feature distributions, shipment patterns, and model performance.
"""))


# ============================================================
# SECTION 18 — CONCLUSION
# ============================================================

cells.append(md("""## 17. Conclusion <a id="conclusion"></a>

This project developed an end-to-end supply chain analytics system using historical DataCo shipment records. Two complementary machine learning tasks were implemented:

1. **Regression** to estimate actual delivery time.
2. **Classification** to estimate late-delivery risk.

The modelling pipeline excludes the identified outcome-related leakage variables and evaluates the models using a chronological train/test split.

### Key Technical Achievements

- Built a reproducible end-to-end machine learning pipeline.
- Applied leakage controls to the modelling feature set.
- Used 31 final modelling features.
- Evaluated multiple regression algorithms.
- Evaluated multiple classification algorithms.
- Achieved **RMSE = 1.259 days** with the selected Random Forest regression model.
- Achieved **F1 = 0.695 and ROC-AUC = 0.768** with the selected XGBoost classification model.
- Applied SHAP-based explainability to investigate feature contributions.
- Built an interactive Streamlit dashboard for analysis and prediction.
- Created a prediction interface using the saved model artifacts.

### Key Business Findings

1. Approximately 54.8% of shipments in the dataset are classified as late.
2. Standard Class accounts for the highest absolute volume of late deliveries because of its large shipment volume.
3. The available pre-delivery features contain useful predictive signal for delivery-risk classification.
4. Same Day shipping has the lowest observed late-delivery rate among the shipping modes in this dataset.
5. Scheduled delivery days is among the strongest predictive features identified in the model analysis.

### Limitations

- The source dataset covers 2015–2018, so current supply-chain conditions may differ.
- The regression R² of approximately 0.39 indicates substantial unexplained variation in delivery time.
- Additional information such as carrier performance, weather, traffic, inventory availability, and operational disruptions is not included in the current dataset.
- SHAP values describe associations with model predictions and do not establish causation.
- Probability thresholds require operational calibration before being used for real-world automated decisions.
- The current saved prediction artifacts expect the project's preprocessed numeric feature representation rather than arbitrary raw categorical order records.
- Historical test-set performance does not guarantee future production performance.

### Future Work

- Integrate real-time order-management data.
- Add carrier-level performance features.
- Incorporate external information such as weather and traffic.
- Investigate temporal forecasting for shipment volume.
- Add automated model monitoring and data-drift detection.
- Calibrate classification thresholds using operational cost and intervention capacity.
- Validate model performance on newer and independent datasets.
- Develop a complete raw-input preprocessing pipeline for production inference.

---

**Author:** K. Niveditha Sai  
**Programme:** IBM SkillsBuild Data Analytics with AI Internship  
**Year:** 2026
"""))


# ============================================================
# ASSEMBLE NOTEBOOK
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


with open(
    NB_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        nb,
        f,
        indent=1,
        ensure_ascii=False
    )


print(
    f"Notebook written: {NB_PATH}"
)

print(
    f"Total cells: {len(cells)}"
)