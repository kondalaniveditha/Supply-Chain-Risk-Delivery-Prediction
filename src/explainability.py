"""
src/explainability.py
=====================
SHAP-based model explainability utilities.

Usage notes
-----------
- TreeExplainer is used for tree-based models (RF, GBM, XGBoost).
- LinearExplainer is used for linear models.
- Explanations are computed on a sample of the test set for speed.
- Causal language is avoided: use "important predictive factor" /
  "associated with model predictions" rather than "causes".
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def get_shap_explainer(model, X_background: np.ndarray):
    """Return the appropriate SHAP explainer for the given model type."""
    model_type = type(model).__name__
    if model_type in ("RandomForestRegressor", "RandomForestClassifier",
                      "GradientBoostingRegressor", "GradientBoostingClassifier",
                      "XGBRegressor", "XGBClassifier"):
        return shap.TreeExplainer(model)
    elif model_type in ("LinearRegression", "LogisticRegression"):
        return shap.LinearExplainer(model, X_background)
    else:
        # Fallback: KernelExplainer (slower)
        logger.warning("Using KernelExplainer for %s – this may be slow.", model_type)
        return shap.KernelExplainer(model.predict, shap.sample(X_background, 100))


def compute_shap_values(model,
                        X: np.ndarray,
                        X_background: np.ndarray,
                        sample_size: int = 2000) -> np.ndarray:
    """Compute SHAP values on a random sample of X.

    Returns
    -------
    shap_values : np.ndarray of shape (sample_size, n_features)
    X_sample    : np.ndarray of the sampled rows
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    explainer   = get_shap_explainer(model, X_background)
    shap_vals   = explainer.shap_values(X_sample)

    # For binary classifiers that return list of two arrays, take class-1 array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return shap_vals, X_sample


def plot_shap_summary(shap_values: np.ndarray,
                      X_sample: np.ndarray,
                      feature_names: list[str],
                      title: str = "SHAP Summary Plot",
                      save_path: str | None = None) -> None:
    """Beeswarm SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved: %s", save_path)
    plt.close()


def plot_shap_bar(shap_values: np.ndarray,
                  feature_names: list[str],
                  title: str = "SHAP Mean |Value| – Global Feature Importance",
                  top_n: int = 15,
                  save_path: str | None = None) -> None:
    """Horizontal bar chart of mean absolute SHAP values."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#3b82d4")
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP bar plot saved: %s", save_path)
    plt.close()
    return imp_df.sort_values("importance", ascending=False)


def get_top_shap_features(shap_values: np.ndarray,
                          feature_names: list[str],
                          top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of top-N features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df
