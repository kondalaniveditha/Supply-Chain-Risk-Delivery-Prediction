"""
src/regression_models.py
========================
Train and compare regression models for delivery-time prediction.

Target : Days for shipping (real)
Models : DummyRegressor (baseline), LinearRegression, RandomForestRegressor,
         GradientBoostingRegressor, XGBRegressor
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_regression_models() -> dict:
    """Return a dict of {name: unfitted_estimator}."""
    return {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_features: bool = True,
) -> tuple[dict, StandardScaler | None]:
    """Fit all regression models on training data.

    Parameters
    ----------
    X_train       : Feature matrix (numeric, no missing values).
    y_train       : Target (Days for shipping real).
    scale_features: If True, apply StandardScaler to numeric features.

    Returns
    -------
    (fitted_models_dict, scaler_or_None)
    """
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        scaler = None
        X_scaled = X_train.values

    models = get_regression_models()
    fitted = {}
    for name, model in models.items():
        logger.info("Training regression model: %s", name)
        model.fit(X_scaled, y_train)
        fitted[name] = model

    return fitted, scaler


def save_regression_artifacts(
    model,
    scaler,
    feature_names: list[str],
    model_dir: str = "models",
    model_name: str = "best_regression_model",
) -> None:
    """Persist the best model, scaler, and feature names."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,         os.path.join(model_dir, f"{model_name}.joblib"))
    joblib.dump(scaler,        os.path.join(model_dir, f"{model_name}_scaler.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, f"{model_name}_features.joblib"))
    logger.info("Saved best regression model to models/%s.joblib", model_name)


def load_regression_artifacts(
    model_dir: str = "models",
    model_name: str = "best_regression_model",
) -> tuple:
    """Load a saved regression model, scaler, and feature names."""
    model         = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))
    scaler        = joblib.load(os.path.join(model_dir, f"{model_name}_scaler.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, f"{model_name}_features.joblib"))
    return model, scaler, feature_names
