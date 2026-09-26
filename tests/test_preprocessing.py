"""
tests/test_preprocessing.py
============================
Unit tests for src/data_preprocessing.py

Run from project root:
    venv/Scripts/python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    LEAKAGE_COLS,
    PII_COLS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
)

RAW_PATH = "data/raw/DataCoSupplyChainDataset.csv"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Load a small slice of the raw dataset."""
    if not os.path.exists(RAW_PATH):
        pytest.skip("Raw dataset not found at data/raw/DataCoSupplyChainDataset.csv")
    return load_raw_data(RAW_PATH)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    """Run the full preprocessing chain on the raw data."""
    df = parse_dates(raw_df)
    df = create_datetime_features(df)
    df = drop_leakage_and_pii(df)
    df = handle_missing_values(df)
    return df


# -----------------------------------------------------------------------
# Dataset loading tests
# -----------------------------------------------------------------------

class TestDataLoading:
    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_row_count(self, raw_df):
        assert len(raw_df) == 180_519, f"Expected 180519 rows, got {len(raw_df)}"

    def test_expected_column_count(self, raw_df):
        assert raw_df.shape[1] == 53, f"Expected 53 columns, got {raw_df.shape[1]}"

    def test_target_columns_present(self, raw_df):
        assert TARGET_REGRESSION in raw_df.columns
        assert TARGET_CLASSIFICATION in raw_df.columns

    def test_target_regression_range(self, raw_df):
        assert raw_df[TARGET_REGRESSION].min() >= 0
        assert raw_df[TARGET_REGRESSION].max() <= 10  # reasonable upper bound

    def test_target_classification_binary(self, raw_df):
        assert set(raw_df[TARGET_CLASSIFICATION].unique()).issubset({0, 1})


# -----------------------------------------------------------------------
# Data quality tests
# -----------------------------------------------------------------------

class TestDataQuality:
    def test_quality_report_shape(self, raw_df):
        qr = report_data_quality(raw_df)
        assert qr.shape[0] == raw_df.shape[1]
        assert "missing_pct" in qr.columns

    def test_no_duplicates(self, raw_df):
        assert raw_df.duplicated().sum() == 0

    def test_product_description_fully_missing(self, raw_df):
        assert raw_df["Product Description"].isnull().all()


# -----------------------------------------------------------------------
# Date parsing tests
# -----------------------------------------------------------------------

class TestDateParsing:
    def test_order_date_parsed(self, raw_df):
        df = parse_dates(raw_df)
        assert "order_date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"])

    def test_original_date_string_dropped(self, raw_df):
        df = parse_dates(raw_df)
        assert "order date (DateOrders)" not in df.columns

    def test_datetime_features_created(self, raw_df):
        df = parse_dates(raw_df)
        df = create_datetime_features(df)
        for col in ["order_year", "order_month", "order_quarter", "order_day",
                    "order_dayofweek", "order_hour"]:
            assert col in df.columns, f"Missing engineered column: {col}"

    def test_order_month_valid_range(self, raw_df):
        df = parse_dates(raw_df)
        df = create_datetime_features(df)
        assert df["order_month"].between(1, 12).all()

    def test_order_dayofweek_valid_range(self, raw_df):
        df = parse_dates(raw_df)
        df = create_datetime_features(df)
        assert df["order_dayofweek"].between(0, 6).all()


# -----------------------------------------------------------------------
# Leakage and PII removal tests
# -----------------------------------------------------------------------

class TestLeakageAndPII:
    def test_leakage_columns_removed(self, processed_df):
        for col in LEAKAGE_COLS:
            assert col not in processed_df.columns, f"Leakage column still present: {col}"

    def test_pii_columns_removed(self, processed_df):
        for col in PII_COLS:
            assert col not in processed_df.columns, f"PII column still present: {col}"

    def test_shipping_date_removed(self, processed_df):
        assert "shipping date (DateOrders)" not in processed_df.columns

    def test_targets_still_present(self, processed_df):
        """Targets should still be present after leakage/PII removal."""
        assert TARGET_REGRESSION in processed_df.columns
        assert TARGET_CLASSIFICATION in processed_df.columns


# -----------------------------------------------------------------------
# Missing value handling tests
# -----------------------------------------------------------------------

class TestMissingValueHandling:
    def test_no_missing_after_imputation(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        assert df_clean.isnull().sum().sum() == 0, "Missing values remain after imputation"


# -----------------------------------------------------------------------
# Feature matrix tests
# -----------------------------------------------------------------------

class TestFeatureMatrix:
    def test_feature_matrix_no_targets(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        df_enc, _ = encode_categoricals(df_clean, fit=True)
        X, y_reg, y_clf = build_feature_matrix(df_enc)
        assert TARGET_REGRESSION not in X.columns
        assert TARGET_CLASSIFICATION not in X.columns

    def test_targets_correct_type(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        df_enc, _ = encode_categoricals(df_clean, fit=True)
        X, y_reg, y_clf = build_feature_matrix(df_enc)
        assert y_reg.dtype in [np.int64, np.float64]
        assert set(y_clf.unique()).issubset({0, 1})

    def test_no_missing_in_features(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        df_enc, _ = encode_categoricals(df_clean, fit=True)
        X, _, _ = build_feature_matrix(df_enc)
        num_X = X.select_dtypes(include=[np.number])
        assert num_X.isnull().sum().sum() == 0


# -----------------------------------------------------------------------
# Chronological split tests
# -----------------------------------------------------------------------

class TestChronologicalSplit:
    def test_split_sizes(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        train, test = chronological_split(df_clean, test_frac=0.20)
        total = len(train) + len(test)
        assert total == len(df_clean)
        assert len(test) == pytest.approx(len(df_clean) * 0.20, rel=0.05)

    def test_train_before_test_chronologically(self, processed_df):
        df_clean = handle_missing_values(processed_df)
        train, test = chronological_split(df_clean, test_frac=0.20)
        if "order_date" in train.columns and "order_date" in test.columns:
            max_train_date = train["order_date"].max()
            min_test_date  = test["order_date"].min()
            assert max_train_date <= min_test_date, \
                "Training data contains dates after test data (data leakage via split)"
