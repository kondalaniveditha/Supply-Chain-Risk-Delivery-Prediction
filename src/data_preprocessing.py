"""
src/data_preprocessing.py
==========================
Reusable preprocessing pipeline for the DataCo Supply Chain dataset.

Design principles
-----------------
* Raw data is NEVER modified (read-only source).
* All post-delivery / leakage columns are removed BEFORE any modelling.
* PII columns are dropped early.
* The processed output is deterministic and reproducible.

Leakage policy
--------------
The following columns are considered leakage because they become known
only AFTER the shipment outcome and must not be used as model features:
  - Delivery Status
  - Days for shipping (real)      ← regression TARGET
  - Late_delivery_risk            ← classification TARGET
  - shipping date (DateOrders)
They are removed from the feature matrix (but kept in a separate
targets frame and in the full processed file for EDA purposes).
"""

import os
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Column role definitions
# ---------------------------------------------------------------------------

# Post-delivery / leakage – must NEVER enter feature matrix
LEAKAGE_COLS = [
    "Delivery Status",
    "shipping date (DateOrders)",
]

# Regression target
TARGET_REGRESSION = "Days for shipping (real)"

# Classification target
TARGET_CLASSIFICATION = "Late_delivery_risk"

# PII columns – removed entirely
PII_COLS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
]

# Meaningless identifiers
IDENTIFIER_COLS = [
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Cardprod Id",
    "Order Item Id",
    "Product Card Id",
    "Product Category Id",
    "Department Id",
    "Category Id",
]

# Sparse / low-signal columns
SPARSE_COLS = [
    "Product Description",   # 100 % missing
    "Product Image",         # URL – no predictive signal
    "Order Zipcode",         # ~86 % missing
    "Order State",           # high cardinality, mostly redundant with Order Region
    "Customer State",        # PII-adjacent, redundant
    "Customer City",         # high cardinality
    "Order City",            # high cardinality
    "Order Country",         # redundant with Market
    "Customer Country",      # redundant with Market
    "Latitude",              # continuous geo – noisy proxy
    "Longitude",
]

# Categorical features to encode
CATEGORICAL_FEATURES = [
    "Type",
    "Customer Segment",
    "Market",
    "Order Region",
    "Shipping Mode",
    "Department Name",
    "Category Name",
    "Order Status",
    "Product Status",
]

# Numerical features
NUMERICAL_FEATURES = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(raw_path: str = "data/raw/DataCoSupplyChainDataset.csv") -> pd.DataFrame:
    """Load the raw DataCo dataset and return a DataFrame.

    The CSV uses latin-1 encoding (confirmed empirically).
    """
    logger.info("Loading raw data from: %s", raw_path)
    df = pd.read_csv(raw_path, encoding="latin1", low_memory=False)
    logger.info("Raw shape: %s rows × %s cols", *df.shape)
    return df


def report_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return a quality-summary DataFrame (missing %, unique counts, dtypes)."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
        "unique_count": df.nunique(),
        "sample_value": df.iloc[0],
    })
    return summary


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse order date column; shipping date is leakage – dropped elsewhere."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(
        df["order date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    # Drop the original string column
    df = df.drop(columns=["order date (DateOrders)"])
    return df


def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create legitimate pre-delivery date/time features from order_date."""
    df = df.copy()
    df["order_year"]      = df["order_date"].dt.year
    df["order_month"]     = df["order_date"].dt.month
    df["order_quarter"]   = df["order_date"].dt.quarter
    df["order_day"]       = df["order_date"].dt.day
    df["order_dayofweek"] = df["order_date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["order_hour"]      = df["order_date"].dt.hour
    return df


def drop_leakage_and_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage, PII, identifier, and sparse columns.

    The TARGET columns are NOT dropped here so EDA can still use them;
    they are separated in `build_feature_matrix`.
    """
    df = df.copy()
    cols_to_drop = (
        LEAKAGE_COLS
        + PII_COLS
        + IDENTIFIER_COLS
        + SPARSE_COLS
        + ["shipping date (DateOrders)"]   # leakage – future info
    )
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped %d leakage/PII/identifier/sparse columns.", len(cols_to_drop))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop remaining missing values."""
    df = df.copy()
    # Customer Lname (8 missing) – low signal, already removed as PII/sparse
    # Any remaining categoricals → fill with 'Unknown'
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna("Unknown")
            logger.info("Filled %d missing in '%s' with 'Unknown'.", n_missing, col)
    # Numerical – fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled %d missing in '%s' with median %.4f.", n_missing, col, median_val)
    return df


def encode_categoricals(df: pd.DataFrame,
                        encoders: dict | None = None,
                        fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical features.

    Parameters
    ----------
    df       : DataFrame with categorical columns already cleaned.
    encoders : Existing LabelEncoder dict (used during inference).
    fit      : If True, fit new encoders and return them.

    Returns
    -------
    (encoded_df, encoders_dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate features, regression target, and classification target.

    Returns
    -------
    X  : feature DataFrame (no targets, no leakage)
    y_reg  : Days for shipping (real)
    y_clf  : Late_delivery_risk
    """
    y_reg = df[TARGET_REGRESSION].copy()
    y_clf = df[TARGET_CLASSIFICATION].copy()

    drop_cols = [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    # Also drop the parsed date column (not useful as a raw feature)
    if "order_date" in df.columns:
        drop_cols.append("order_date")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y_reg, y_clf


def chronological_split(df: pd.DataFrame,
                        test_frac: float = 0.2
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically: earlier rows → train, later → test.

    This avoids leaking future information into the training set.
    """
    df_sorted = df.sort_values("order_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:split_idx].copy()
    test  = df_sorted.iloc[split_idx:].copy()
    logger.info("Train: %d rows | Test: %d rows (chronological split).", len(train), len(test))
    return train, test


def run_full_pipeline(
    raw_path: str = "data/raw/DataCoSupplyChainDataset.csv",
    processed_path: str = "data/processed/supply_chain_processed.csv",
    test_frac: float = 0.20,
) -> dict:
    """Execute the complete preprocessing pipeline end-to-end.

    Returns a dict with keys:
      df_full, df_train, df_test,
      X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test,
      encoders, quality_report
    """
    # 1. Load
    df_raw = load_raw_data(raw_path)

    # 2. Quality report (before cleaning)
    quality_report = report_data_quality(df_raw)

    # 3. Parse dates
    df = parse_dates(df_raw)

    # 4. Create datetime features (legitimate pre-delivery info)
    df = create_datetime_features(df)

    # 5. Drop leakage, PII, identifiers, sparse columns
    df = drop_leakage_and_pii(df)

    # 6. Handle missing values
    df = handle_missing_values(df)

    # 7. Chronological split (before encoding to prevent data leakage through encoders)
    df_train, df_test = chronological_split(df, test_frac=test_frac)

    # 8. Encode categoricals (fit on train, transform both)
    df_train_enc, encoders = encode_categoricals(df_train, fit=True)
    df_test_enc, _         = encode_categoricals(df_test, encoders=encoders, fit=False)

    # 9. Build feature matrices
    X_train, y_reg_train, y_clf_train = build_feature_matrix(df_train_enc)
    X_test,  y_reg_test,  y_clf_test  = build_feature_matrix(df_test_enc)

    # 10. Save processed full dataset (for EDA / dashboard)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    # Save the encoded full df (needed for dashboard)
    df_full_enc, _ = encode_categoricals(df, encoders=encoders, fit=False)
    df_full_enc.to_csv(processed_path, index=False)
    logger.info("Processed data saved to: %s", processed_path)

    # Also save train/test splits
    train_path = processed_path.replace(".csv", "_train.csv")
    test_path  = processed_path.replace(".csv", "_test.csv")
    df_train_enc.to_csv(train_path, index=False)
    df_test_enc.to_csv(test_path, index=False)
    logger.info("Train split saved to: %s", train_path)
    logger.info("Test  split saved to: %s", test_path)

    return {
        "df_full": df,
        "df_full_enc": df_full_enc,
        "df_train": df_train_enc,
        "df_test": df_test_enc,
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train,
        "y_clf_test": y_clf_test,
        "encoders": encoders,
        "quality_report": quality_report,
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print("Feature columns:", result["X_train"].columns.tolist())
    print("Train shape:", result["X_train"].shape)
    print("Test shape:", result["X_test"].shape)
