"""
src/evaluation.py
=================
Model evaluation utilities for regression and classification tasks.
All metrics are computed from actual model predictions – never hard-coded.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for a regression model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def compare_regression_models(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Return a comparison DataFrame for all regression models.

    Parameters
    ----------
    fitted_models : {name: fitted_estimator}
    X_test        : Scaled test feature matrix.
    y_test        : True targets.

    Returns
    -------
    DataFrame with columns [Model, MAE, RMSE, R²]
    """
    rows = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        metrics["Model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows)[["Model", "MAE", "RMSE", "R2"]].rename(columns={"R2": "R²"})
    df = df.sort_values("RMSE").reset_index(drop=True)
    return df


def select_best_regression_model(comparison_df: pd.DataFrame,
                                 fitted_models: dict,
                                 metric: str = "RMSE") -> tuple[str, object]:
    """Select the best regression model by lowest RMSE (default)."""
    best_name = comparison_df.sort_values(metric).iloc[0]["Model"]
    return best_name, fitted_models[best_name]


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC."""
    metrics = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        metrics["PR-AUC"]  = round(average_precision_score(y_true, y_prob), 4)
    return metrics


def compare_classification_models(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Return a comparison DataFrame for all classification models."""
    rows = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = evaluate_classification(y_test, y_pred, y_prob)
        metrics["Model"] = name
        rows.append(metrics)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def select_best_classification_model(comparison_df: pd.DataFrame,
                                     fitted_models: dict,
                                     metric: str = "F1") -> tuple[str, object]:
    """Select the best classification model by highest F1 (default)."""
    best_name = comparison_df.sort_values(metric, ascending=False).iloc[0]["Model"]
    return best_name, fitted_models[best_name]


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix as a 2-D numpy array."""
    return confusion_matrix(y_true, y_pred)
