# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project

Python 3.13 ML project — supply chain delivery time regression + late-delivery risk classification.
Dataset: DataCo Supply Chain CSV (180,519 rows × 53 cols) at `data/raw/DataCoSupplyChainDataset.csv` (latin-1 encoding).

## Commands

```bash
# Train all models + generate all figures (run from project root)
venv\Scripts\python scripts/train_and_evaluate.py

# Run all tests
venv\Scripts\python -m pytest tests/ -v

# Run a single test
venv\Scripts\python -m pytest tests/test_models.py::TestPredictionPipeline::test_risk_category_assignment -v

# Launch dashboard
venv\Scripts\streamlit run dashboard/app.py

# Regenerate notebook
venv\Scripts\python scripts/build_notebook.py

# Regenerate DOCX report
venv\Scripts\python scripts/build_report.py

# Install dependencies
venv\Scripts\pip install -r requirements.txt
```

## Critical Constraints (Non-Obvious)

- **Leakage policy is enforced in `src/data_preprocessing.py`**: `LEAKAGE_COLS`, `PII_COLS`, `IDENTIFIER_COLS`, `SPARSE_COLS` lists define what is removed. Never add `Delivery Status`, `shipping date (DateOrders)`, or either target column to features.
- **Train/test split is chronological**, not random — `chronological_split()` sorts by `order_date` before splitting. Adding random shuffling before the split breaks this.
- **Scaler is fit on train only** — `encode_categoricals()` and `StandardScaler` are always fit on training data and applied (not re-fit) to test data.
- **CSV encoding is latin-1** — loading with default UTF-8 will fail on this dataset.
- **`pandas 3.x` breaking change**: `select_dtypes(include="object")` raises a deprecation warning (use `include="str"` going forward). Current code uses `"object"` to maintain pandas 2/3 compatibility.
- **`matplotlib.use("Agg")`** must be called before any other matplotlib import in non-notebook scripts — already done in `scripts/train_and_evaluate.py` and `src/explainability.py`.
- **Models are saved to `models/` via `joblib`**. Tests skip model tests if `.joblib` files are missing (run `scripts/train_and_evaluate.py` first).
- **SHAP `TreeExplainer` returns a list for binary classifiers** — `src/explainability.py` handles this by taking `shap_vals[1]` (class-1 array).
- **Feature columns** are saved as `models/best_*_features.joblib` — the scaler and model MUST use the same column list. `batch_predict()` in `src/prediction.py` uses `.reindex(columns=features, fill_value=0)` to handle column alignment.

## Architecture

```
src/data_preprocessing.py  → run_full_pipeline() → data/processed/
src/feature_engineering.py → add_all_features()
src/regression_models.py   → train_regression_models() → models/
src/classification_models.py → train_classification_models() → models/
src/evaluation.py          → compare_*_models()
src/explainability.py      → compute_shap_values(), plot_shap_*()
src/prediction.py          → predict_single(), batch_predict()
dashboard/app.py           → Streamlit (3 pages, uses raw CSV for display labels)
```

## Risk Category Thresholds (in `src/prediction.py`)

- `< 0.40` → LOW
- `0.40–0.65` → MEDIUM  
- `> 0.65` → HIGH
