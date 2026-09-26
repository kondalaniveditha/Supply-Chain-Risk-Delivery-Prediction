"""
src/prediction.py
=================
Single-shipment prediction pipeline.

Accepts pre-delivery shipment attributes, applies the saved preprocessing
pipeline and returns:
  - predicted_delivery_days  (regression)
  - late_probability         (classification)
  - risk_category            (LOW / MEDIUM / HIGH)

Probability thresholds
----------------------
  < 0.40  → LOW
  0.40–0.65 → MEDIUM
  > 0.65  → HIGH

These thresholds are documented here and in the report.  They represent
a reasonable starting point; the precision-recall tradeoff means:
  - Lower threshold → fewer missed late deliveries (higher recall)
  - Higher threshold → fewer false alarms (higher precision)
Operational teams should adjust based on the cost of false alarms vs
the cost of missed late deliveries.
"""

import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Risk probability thresholds
LOW_THRESHOLD    = 0.40
MEDIUM_THRESHOLD = 0.65


def assign_risk_category(probability: float) -> str:
    """Map a late-delivery probability to a risk category string."""
    if probability < LOW_THRESHOLD:
        return "LOW"
    elif probability < MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def load_pipeline(model_dir: str = "models") -> dict:
    """Load both models, scalers, and feature name lists."""
    import os
    artifacts = {}
    for task in ("regression", "classification"):
        prefix = f"best_{task}_model"
        artifacts[f"{task}_model"]    = joblib.load(os.path.join(model_dir, f"{prefix}.joblib"))
        artifacts[f"{task}_scaler"]   = joblib.load(os.path.join(model_dir, f"{prefix}_scaler.joblib"))
        artifacts[f"{task}_features"] = joblib.load(os.path.join(model_dir, f"{prefix}_features.joblib"))
    return artifacts


def predict_single(input_dict: dict, artifacts: dict) -> dict:
    """Make predictions for a single pre-delivery shipment record.

    Parameters
    ----------
    input_dict : Dict of feature name → value (raw, pre-encoding).
    artifacts  : Loaded pipeline artifacts from load_pipeline().

    Returns
    -------
    dict with keys:
        predicted_delivery_days, late_probability, risk_category
    """
    reg_features = artifacts["regression_features"]
    clf_features = artifacts["classification_features"]

    # Build single-row DataFrames aligned to expected feature columns
    row_reg = pd.DataFrame([{f: input_dict.get(f, 0) for f in reg_features}])
    row_clf = pd.DataFrame([{f: input_dict.get(f, 0) for f in clf_features}])

    # Scale
    X_reg = artifacts["regression_scaler"].transform(row_reg)
    X_clf = artifacts["classification_scaler"].transform(row_clf)

    # Predict
    pred_days = float(artifacts["regression_model"].predict(X_reg)[0])
    pred_days = max(0, round(pred_days, 1))   # delivery days can't be negative

    late_prob = float(
        artifacts["classification_model"].predict_proba(X_clf)[0, 1]
    )
    risk_cat = assign_risk_category(late_prob)

    return {
        "predicted_delivery_days": pred_days,
        "late_probability": round(late_prob, 4),
        "risk_category": risk_cat,
    }


def batch_predict(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Make predictions for a batch DataFrame.

    Returns the input DataFrame with three additional columns appended:
    predicted_delivery_days, late_probability, risk_category.
    """
    reg_features = artifacts["regression_features"]
    clf_features = artifacts["classification_features"]

    X_reg = artifacts["regression_scaler"].transform(
        df[[f for f in reg_features if f in df.columns]].reindex(columns=reg_features, fill_value=0)
    )
    X_clf = artifacts["classification_scaler"].transform(
        df[[f for f in clf_features if f in df.columns]].reindex(columns=clf_features, fill_value=0)
    )

    df = df.copy()
    df["predicted_delivery_days"] = np.clip(
        artifacts["regression_model"].predict(X_reg), 0, None
    ).round(1)
    df["late_probability"] = artifacts["classification_model"].predict_proba(X_clf)[:, 1].round(4)
    df["risk_category"]    = df["late_probability"].apply(assign_risk_category)
    return df
