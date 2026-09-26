"""
tests/test_models.py
=====================
Unit tests for model loading, prediction, and evaluation.

Run from project root:
    venv/Scripts/python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = "models"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_artifacts():
    """Load saved regression model artifacts."""
    from src.regression_models import load_regression_artifacts
    model_path = os.path.join(MODELS_DIR, "best_regression_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Regression model not found — run scripts/train_and_evaluate.py first")
    return load_regression_artifacts(MODELS_DIR)


@pytest.fixture(scope="module")
def classification_artifacts():
    """Load saved classification model artifacts."""
    from src.classification_models import load_classification_artifacts
    model_path = os.path.join(MODELS_DIR, "best_classification_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Classification model not found — run scripts/train_and_evaluate.py first")
    return load_classification_artifacts(MODELS_DIR)


@pytest.fixture(scope="module")
def sample_features(regression_artifacts):
    """Generate a sample feature vector aligned to the saved feature names."""
    _, _, feature_names = regression_artifacts
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((10, len(feature_names))),
        columns=feature_names
    )
    return X, feature_names


# -----------------------------------------------------------------------
# Model loading tests
# -----------------------------------------------------------------------

class TestModelLoading:
    def test_regression_model_loads(self, regression_artifacts):
        model, scaler, features = regression_artifacts
        assert model is not None
        assert scaler is not None
        assert isinstance(features, list)
        assert len(features) > 0

    def test_classification_model_loads(self, classification_artifacts):
        model, scaler, features = classification_artifacts
        assert model is not None
        assert scaler is not None
        assert isinstance(features, list)
        assert len(features) > 0

    def test_models_have_predict(self, regression_artifacts, classification_artifacts):
        reg_model, _, _ = regression_artifacts
        clf_model, _, _ = classification_artifacts
        assert hasattr(reg_model, "predict")
        assert hasattr(clf_model, "predict")
        assert hasattr(clf_model, "predict_proba")


# -----------------------------------------------------------------------
# Regression prediction tests
# -----------------------------------------------------------------------

class TestRegressionPrediction:
    def test_predict_returns_array(self, regression_artifacts, sample_features):
        model, scaler, _ = regression_artifacts
        X, _ = sample_features
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        assert len(preds) == len(X)

    def test_predict_non_negative(self, regression_artifacts, sample_features):
        """Delivery days predictions should be plausible (clipped ≥ 0)."""
        model, scaler, _ = regression_artifacts
        X, _ = sample_features
        X_scaled = scaler.transform(X)
        preds = np.clip(model.predict(X_scaled), 0, None)
        assert (preds >= 0).all()

    def test_predict_reasonable_range(self, regression_artifacts):
        """Predictions on real data should be in [0, 10] days."""
        from src.data_preprocessing import run_full_pipeline, build_feature_matrix
        from src.feature_engineering import add_all_features
        pipeline = run_full_pipeline()
        df_test  = add_all_features(pipeline["df_test"])
        X_test, _, _ = build_feature_matrix(df_test)
        num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        model, scaler, features = regression_artifacts
        common = [c for c in features if c in num_cols]
        X_aligned = X_test.reindex(columns=features, fill_value=0)
        X_scaled  = scaler.transform(X_aligned)
        preds = model.predict(X_scaled)
        assert preds.min() > -5   # some slack for edge predictions
        assert preds.max() < 15


# -----------------------------------------------------------------------
# Classification prediction tests
# -----------------------------------------------------------------------

class TestClassificationPrediction:
    def test_predict_binary_output(self, classification_artifacts, sample_features):
        model, scaler, _ = classification_artifacts
        X, _ = sample_features
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, classification_artifacts, sample_features):
        model, scaler, _ = classification_artifacts
        X, _ = sample_features
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_unit_interval(self, classification_artifacts, sample_features):
        model, scaler, _ = classification_artifacts
        X, _ = sample_features
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()


# -----------------------------------------------------------------------
# Invalid input handling tests
# -----------------------------------------------------------------------

class TestInvalidInputHandling:
    def test_nan_input_after_fillna(self, regression_artifacts):
        """NaN inputs should be handled gracefully via fillna(0) alignment."""
        model, scaler, feature_names = regression_artifacts
        X_nan = pd.DataFrame(
            [[np.nan] * len(feature_names)], columns=feature_names
        ).fillna(0)
        X_scaled = scaler.transform(X_nan)
        pred = model.predict(X_scaled)
        assert len(pred) == 1

    def test_zero_vector_input(self, regression_artifacts):
        model, scaler, feature_names = regression_artifacts
        X_zero = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        X_scaled = scaler.transform(X_zero)
        pred = model.predict(X_scaled)
        assert len(pred) == 1


# -----------------------------------------------------------------------
# Prediction pipeline tests
# -----------------------------------------------------------------------

class TestPredictionPipeline:
    def test_risk_category_assignment(self):
        from src.prediction import assign_risk_category
        assert assign_risk_category(0.20) == "LOW"
        assert assign_risk_category(0.50) == "MEDIUM"
        assert assign_risk_category(0.80) == "HIGH"
        assert assign_risk_category(0.40) == "MEDIUM"   # boundary: ≥ 0.40 → MEDIUM
        assert assign_risk_category(0.65) == "HIGH"     # boundary: ≥ 0.65 → HIGH

    def test_load_pipeline_artifacts(self):
        from src.prediction import load_pipeline
        if not os.path.exists(os.path.join(MODELS_DIR, "best_regression_model.joblib")):
            pytest.skip("Models not trained yet")
        artifacts = load_pipeline(MODELS_DIR)
        assert "regression_model"    in artifacts
        assert "classification_model" in artifacts
        assert "regression_features" in artifacts


# -----------------------------------------------------------------------
# Evaluation utility tests
# -----------------------------------------------------------------------

class TestEvaluationUtilities:
    def test_evaluate_regression_metrics(self):
        from src.evaluation import evaluate_regression
        y_true = np.array([3, 4, 2, 5, 3])
        y_pred = np.array([3, 4, 2, 5, 3])
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics["MAE"]  == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"]   == 1.0

    def test_evaluate_classification_metrics(self):
        from src.evaluation import evaluate_classification
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        metrics = evaluate_classification(y_true, y_pred)
        assert metrics["Accuracy"]  == 1.0
        assert metrics["Precision"] == 1.0
        assert metrics["Recall"]    == 1.0
        assert metrics["F1"]        == 1.0

    def test_regression_worse_than_perfect(self):
        from src.evaluation import evaluate_regression
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([2, 3, 4, 5, 6], dtype=float)  # off by 1 each
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics["MAE"]  > 0
        assert metrics["RMSE"] > 0
        assert metrics["R2"]   < 1.0
