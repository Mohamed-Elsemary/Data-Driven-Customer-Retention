"""Integration tests — verify pipeline stages compose correctly.

Covers:
  - Full pipeline (raw data → predictions)
  - NaN non-propagation through stages
  - Preprocessing determinism (no training-serving skew)
  - Business metrics integration with model output
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import feature_engineering
from data_cleaning import clean
from modeling import _compute_business_metrics


class TestFullPipeline:
    """End-to-end pipeline from raw data to model predictions."""

    @pytest.fixture(autouse=True)
    def _patch_test_size(self, monkeypatch):
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

    def test_raw_to_predictions(self, raw_df):
        """Run clean → encode → engineer → split → train → predict
        and verify predictions are sane."""
        df = clean(raw_df)
        df = feature_engineering.encode(df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        assert set(preds).issubset({0, 1}), "Predictions must be binary"
        assert all(0 <= p <= 1 for p in probs), "Probabilities out of [0, 1]"
        assert not np.any(np.isnan(preds)), "NaN in predictions"

    def test_no_nan_propagation(self, raw_df):
        """NaN values must not silently propagate through any stage."""
        df = clean(raw_df)
        df = feature_engineering.encode(df)
        assert df.isnull().sum().sum() == 0, "NaN after encode"

        df = feature_engineering.add_engineered_features(df)
        assert df.isnull().sum().sum() == 0, "NaN after feature engineering"

        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)
        assert X_train.isnull().sum().sum() == 0, "NaN in X_train"
        assert X_test.isnull().sum().sum() == 0, "NaN in X_test"

    def test_all_features_numeric_after_preprocessing(self, raw_df):
        """After full preprocessing, every feature column must be numeric."""
        df = clean(raw_df)
        df = feature_engineering.encode(df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)

        for col in X_train.columns:
            assert pd.api.types.is_numeric_dtype(X_train[col]), (
                f"X_train['{col}'] is {X_train[col].dtype}, not numeric"
            )


class TestDeterminism:
    """Verify preprocessing is deterministic (no training-serving skew)."""

    def test_encode_deterministic(self, raw_df):
        """Running encode() twice on the same input gives identical output."""
        df = clean(raw_df)
        first = feature_engineering.encode(df)
        second = feature_engineering.encode(df)
        pd.testing.assert_frame_equal(first, second)

    def test_split_reproducible(self, featured_df, monkeypatch):
        """Same RANDOM_STATE must produce identical splits."""
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)
        X1, _, y1, _ = feature_engineering.split_and_encode(featured_df)
        X2, _, y2, _ = feature_engineering.split_and_encode(featured_df)
        pd.testing.assert_frame_equal(
            X1.reset_index(drop=True), X2.reset_index(drop=True)
        )
        pd.testing.assert_series_equal(
            y1.reset_index(drop=True), y2.reset_index(drop=True)
        )


class TestBusinessMetricsIntegration:
    """Verify business metrics produce valid output when wired to model predictions."""

    def test_metrics_from_pipeline(self, raw_df, monkeypatch):
        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(raw_df)
        df = feature_engineering.encode(df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = _compute_business_metrics(
            y_test.values, preds, X_test["MonthlyCharges"].values
        )

        expected_keys = {
            "Revenue_Saved", "Net_Retention_Value",
            "Cost_Per_Detection", "Revenue_Leakage",
        }
        assert set(metrics.keys()) == expected_keys
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float"
            assert not np.isnan(val), f"{key} is NaN"
            assert not np.isinf(val), f"{key} is Inf"
