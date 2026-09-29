"""
src/feature_engineering.py
===========================
Additional feature engineering beyond basic preprocessing.
All features created here use only information available PRE-DELIVERY.
"""

import numpy as np
import pandas as pd


def add_schedule_deviation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether scheduled days <= 2 (expedited shipping schedule)."""
    df = df.copy()
    if "Days for shipment (scheduled)" in df.columns:
        df["is_expedited_schedule"] = (df["Days for shipment (scheduled)"] <= 2).astype(int)
    return df


def add_profit_margin_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Profit margin = Order Profit Per Order / Order Item Total (pre-delivery revenue info)."""
    df = df.copy()
    if "Order Profit Per Order" in df.columns and "Order Item Total" in df.columns:
        denom = df["Order Item Total"].replace(0, np.nan)
        df["profit_margin_ratio"] = df["Order Profit Per Order"] / denom
        df["profit_margin_ratio"] = df["profit_margin_ratio"].fillna(0).clip(-5, 5)
    return df


def add_discount_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: is there a non-zero item discount?"""
    df = df.copy()
    if "Order Item Discount" in df.columns:
        df["has_discount"] = (df["Order Item Discount"] > 0).astype(int)
    return df


def add_high_value_order_flag(df: pd.DataFrame, threshold: float = 200.0) -> pd.DataFrame:
    """Flag orders where product price exceeds a threshold."""
    df = df.copy()
    if "Product Price" in df.columns:
        df["is_high_value"] = (df["Product Price"] >= threshold).astype(int)
    return df


def add_weekend_order_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag orders placed on weekends (dayofweek 5=Sat, 6=Sun)."""
    df = df.copy()
    if "order_dayofweek" in df.columns:
        df["is_weekend_order"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return augmented DataFrame."""
    df = add_schedule_deviation_flag(df)
    df = add_profit_margin_ratio(df)
    df = add_discount_flag(df)
    df = add_high_value_order_flag(df)
    df = add_weekend_order_flag(df)
    return df


def get_final_feature_columns(df: pd.DataFrame,
                               exclude: list[str] | None = None) -> list[str]:
    """Return list of numeric columns suitable for modelling."""
    exclude = exclude or []
    # Keep only numeric columns that are not targets / dates
    always_exclude = [
        "Days for shipping (real)",
        "Late_delivery_risk",
        "order_date",
    ] + exclude
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in always_exclude
    ]
    return cols
