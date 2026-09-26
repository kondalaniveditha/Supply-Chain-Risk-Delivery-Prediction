"""
src/classification_models.py
=============================
Train and compare classification models for delivery-delay-risk prediction.

Target : Late_delivery_risk  (0 = on time, 1 = late)
Models : LogisticRegression, RandomForestClassifier,
         GradientBoostingClassifier, XGBClassifier
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_classification_models(class_weight: str | None = "balanced") -> dict:
    """Return a dict of {name: unfitted_estimator}.

    class_weight='balanced' compensates for the slight class imbalance
    (~55 % late, ~45 % on-time) – prevents the model from optimising
    solely for the majority class.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,          # adjust if imbalance severe
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_features: bool = True,
) -> tuple[dict, StandardScaler | None]:
    """Fit all classification models on training data.

    Parameters
    ----------
    X_train       : Feature matrix (numeric, no missing values).
    y_train       : Binary target (0/1).
    scale_features: If True, apply StandardScaler.

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

    models = get_classification_models()
    fitted = {}
    for name, model in models.items():
        logger.info("Training classification model: %s", name)
        model.fit(X_scaled, y_train)
        fitted[name] = model

    return fitted, scaler


def save_classification_artifacts(
    model,
    scaler,
    feature_names: list[str],
    model_dir: str = "models",
    model_name: str = "best_classification_model",
) -> None:
    """Persist the best classifier, scaler, and feature names."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,         os.path.join(model_dir, f"{model_name}.joblib"))
    joblib.dump(scaler,        os.path.join(model_dir, f"{model_name}_scaler.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, f"{model_name}_features.joblib"))
    logger.info("Saved best classification model to models/%s.joblib", model_name)


def load_classification_artifacts(
    model_dir: str = "models",
    model_name: str = "best_classification_model",
) -> tuple:
    """Load a saved classification model, scaler, and feature names."""
    model         = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))
    scaler        = joblib.load(os.path.join(model_dir, f"{model_name}_scaler.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, f"{model_name}_features.joblib"))
    return model, scaler, feature_names
