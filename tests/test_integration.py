"""Integration tests — verify that pipeline stages compose correctly.

Covers (from the course testing framework, Level 3):
  L3-1  Full Pipeline Stages (raw data → prediction with mock data)
  L3-3  Training-Serving Skew (preprocessing consistency)
  L3-4  Feature Computation Consistency
"""

import pytest
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def integration_df():
    """A larger mock dataset (10 rows) for integration tests that
    need enough data for a meaningful train/test split."""
    data = {
        "customerID": [f"INT-{i:04d}" for i in range(10)],
        "gender": ["Male", "Female"] * 5,
        "SeniorCitizen": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes",
                     "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "Yes",
                        "No", "No", "Yes", "No", "Yes"],
        "tenure": [12, 0, 36, 48, 72, 5, 24, 60, 3, 40],
        "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes",
                          "No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": [
            "No", "No phone service", "Yes", "No", "Yes",
            "No phone service", "No", "Yes", "No phone service", "No",
        ],
        "InternetService": [
            "Fiber optic", "DSL", "No", "Fiber optic", "DSL",
            "DSL", "Fiber optic", "No", "DSL", "Fiber optic",
        ],
        "OnlineSecurity": [
            "No", "Yes", "No internet service", "No", "Yes",
            "No", "No", "No internet service", "Yes", "No",
        ],
        "OnlineBackup": [
            "Yes", "No", "No internet service", "No", "Yes",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "DeviceProtection": [
            "No", "Yes", "No internet service", "Yes", "No",
            "No", "Yes", "No internet service", "No", "Yes",
        ],
        "TechSupport": [
            "No", "No", "No internet service", "Yes", "Yes",
            "No", "No", "No internet service", "Yes", "No",
        ],
        "StreamingTV": [
            "Yes", "No", "No internet service", "Yes", "No",
            "Yes", "No", "No internet service", "No", "Yes",
        ],
        "StreamingMovies": [
            "No", "Yes", "No internet service", "No", "Yes",
            "No", "Yes", "No internet service", "Yes", "No",
        ],
        "Contract": [
            "Month-to-month", "One year", "Two year",
            "Month-to-month", "Two year", "Month-to-month",
            "One year", "Two year", "Month-to-month", "One year",
        ],
        "PaperlessBilling": [
            "Yes", "No", "No", "Yes", "No",
            "Yes", "No", "Yes", "No", "Yes",
        ],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Bank transfer (automatic)", "Electronic check",
            "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Electronic check",
        ],
        "MonthlyCharges": [
            70.35, 29.85, 20.05, 99.65, 45.25,
            55.10, 65.40, 19.95, 35.50, 89.90,
        ],
        "TotalCharges": [
            "844.2", "0", "721.8", "4783.2", "3258",
            "275.5", "1569.6", "1197", "106.5", "3596",
        ],
        "Churn": [
            "Yes", "No", "No", "Yes", "No",
            "Yes", "No", "No", "Yes", "No",
        ],
    }
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════
#  L3-1: FULL PIPELINE — RAW DATA → PREDICTIONS
# ═══════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end pipeline from raw data to model predictions."""

    def test_raw_to_predictions(self, integration_df, monkeypatch):
        """Run clean → encode → engineer → split → train → predict
        on mock data and verify the output is sane."""
        from data_cleaning import clean
        import feature_engineering
        from sklearn.linear_model import LogisticRegression

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        # Stage 1: Clean
        df = clean(integration_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

        # Stage 2: Encode
        df = feature_engineering.encode(df)
        assert df.isnull().sum().sum() == 0

        # Stage 3: Engineer features
        df = feature_engineering.add_engineered_features(df)
        assert "_Total_AddOns-Services" in df.columns

        # Stage 4: Split
        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)
        assert len(X_train) + len(X_test) == 10
        assert "Churn" not in X_train.columns

        # Stage 5: Train & Predict
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Prediction sanity
        assert preds.shape == y_test.shape
        assert set(preds).issubset({0, 1}), "Predictions must be binary"
        assert all(0 <= p <= 1 for p in probs), "Probabilities out of [0, 1]"
        assert not np.any(np.isnan(preds)), "NaN in predictions"
        assert not np.any(np.isinf(probs)), "Inf in probabilities"

    def test_pipeline_no_nan_propagation(self, integration_df, monkeypatch):
        """Ensure NaN values do not silently propagate through stages."""
        from data_cleaning import clean
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = feature_engineering.encode(df)
        assert df.isnull().sum().sum() == 0, "NaN after encode"

        df = feature_engineering.add_engineered_features(df)
        assert df.isnull().sum().sum() == 0, "NaN after feature engineering"

        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)
        assert X_train.isnull().sum().sum() == 0, "NaN in X_train"
        assert X_test.isnull().sum().sum() == 0, "NaN in X_test"
        assert y_train.isnull().sum() == 0, "NaN in y_train"
        assert y_test.isnull().sum() == 0, "NaN in y_test"

    def test_pipeline_dtypes_stay_numeric(self, integration_df, monkeypatch):
        """After full preprocessing, all features must be numeric."""
        from data_cleaning import clean
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = feature_engineering.encode(df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)

        for col in X_train.columns:
            assert pd.api.types.is_numeric_dtype(X_train[col]), (
                f"X_train['{col}'] is {X_train[col].dtype}, not numeric"
            )
        for col in X_test.columns:
            assert pd.api.types.is_numeric_dtype(X_test[col]), (
                f"X_test['{col}'] is {X_test[col].dtype}, not numeric"
            )


# ═══════════════════════════════════════════════════════════════
#  L3-3: TRAINING-SERVING SKEW
# ═══════════════════════════════════════════════════════════════

class TestTrainingServingSkew:
    """Verify that preprocessing applied at training time would
    produce identical results if re-applied at serving time."""

    def test_encode_deterministic(self, integration_df):
        """Running encode() on the same input twice gives identical output."""
        from data_cleaning import clean
        from feature_engineering import encode

        df = clean(integration_df)
        first = encode(df)
        second = encode(df)
        pd.testing.assert_frame_equal(first, second)

    def test_feature_engineering_deterministic(self, integration_df):
        """Running add_engineered_features() twice gives identical output."""
        from data_cleaning import clean
        from feature_engineering import encode, add_engineered_features

        df = clean(integration_df)
        df = encode(df)
        first = add_engineered_features(df)
        second = add_engineered_features(df)
        pd.testing.assert_frame_equal(first, second)

    def test_clean_deterministic(self, integration_df):
        """Running clean() twice on the same input gives identical output."""
        from data_cleaning import clean
        first = clean(integration_df)
        second = clean(integration_df)
        pd.testing.assert_frame_equal(first, second)

    def test_single_row_preprocessing_matches_batch(self, integration_df):
        """Preprocessing a single row independently must produce the same
        feature values as processing it within the full batch (for
        stateless transforms).

        Note: This test covers only stateless stages (clean + encode).
        Stateful stages like LoyaltyScore use batch statistics and are
        expected to differ — that's the leakage documented elsewhere.
        """
        from data_cleaning import clean
        from feature_engineering import encode

        df = clean(integration_df)

        # Process full batch
        batch_encoded = encode(df)

        # Process one row alone
        single_row = df.iloc[[0]].copy()
        single_encoded = encode(single_row)

        # Stateless columns should match
        stateless_cols = [
            c for c in single_encoded.columns
            if c in batch_encoded.columns
        ]
        for col in stateless_cols:
            batch_val = batch_encoded.iloc[0][col]
            single_val = single_encoded.iloc[0][col]
            if pd.api.types.is_numeric_dtype(batch_encoded[col]):
                assert batch_val == single_val, (
                    f"Skew detected in '{col}': "
                    f"batch={batch_val}, single={single_val}"
                )


# ═══════════════════════════════════════════════════════════════
#  L3-4: FEATURE COMPUTATION CONSISTENCY
# ═══════════════════════════════════════════════════════════════

class TestFeatureComputationConsistency:
    """Verify features are computed identically across runs."""

    def test_frequency_encoding_consistent_across_runs(
        self, integration_df, monkeypatch
    ):
        """Two split_and_encode() calls with the same seed produce
        identical frequency-encoded Contract values."""
        from data_cleaning import clean
        import feature_engineering
        from feature_engineering import encode, add_engineered_features

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = encode(df)
        df = add_engineered_features(df)

        X_train_1, X_test_1, _, _ = feature_engineering.split_and_encode(df)
        X_train_2, X_test_2, _, _ = feature_engineering.split_and_encode(df)

        pd.testing.assert_series_equal(
            X_train_1["Contract"].reset_index(drop=True),
            X_train_2["Contract"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            X_test_1["Contract"].reset_index(drop=True),
            X_test_2["Contract"].reset_index(drop=True),
        )

    def test_split_reproducibility_with_same_seed(
        self, integration_df, monkeypatch
    ):
        """Same RANDOM_STATE must produce identical splits."""
        from data_cleaning import clean
        import feature_engineering
        from feature_engineering import encode, add_engineered_features

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = encode(df)
        df = add_engineered_features(df)

        X_train_1, X_test_1, y_train_1, y_test_1 = (
            feature_engineering.split_and_encode(df)
        )
        X_train_2, X_test_2, y_train_2, y_test_2 = (
            feature_engineering.split_and_encode(df)
        )

        pd.testing.assert_frame_equal(
            X_train_1.reset_index(drop=True),
            X_train_2.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            X_test_1.reset_index(drop=True),
            X_test_2.reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            y_train_1.reset_index(drop=True),
            y_train_2.reset_index(drop=True),
        )

    def test_business_metrics_integration(self, integration_df, monkeypatch):
        """Verify business metrics produce valid output when integrated
        with model predictions from the full pipeline."""
        from data_cleaning import clean
        import feature_engineering
        from feature_engineering import encode, add_engineered_features
        from modeling import _compute_business_metrics
        from sklearn.linear_model import LogisticRegression

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = encode(df)
        df = add_engineered_features(df)
        X_train, X_test, y_train, y_test = feature_engineering.split_and_encode(df)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = _compute_business_metrics(
            y_test.values, preds, X_test["MonthlyCharges"].values,
        )

        assert isinstance(metrics, dict)
        expected_keys = {
            "Revenue_Saved", "Net_Retention_Value",
            "Cost_Per_Detection", "Revenue_Leakage",
        }
        assert set(metrics.keys()) == expected_keys
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float"
            assert not np.isnan(val), f"{key} is NaN"
            assert not np.isinf(val), f"{key} is Inf"

    def test_correlation_analysis_preserves_features(
        self, integration_df, monkeypatch
    ):
        """correlation_analysis() should drop specific columns but
        not lose any other features."""
        from data_cleaning import clean
        import feature_engineering
        from feature_engineering import (
            encode, add_engineered_features, correlation_analysis,
        )

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.3)

        df = clean(integration_df)
        df = encode(df)
        df = add_engineered_features(df)
        X_train, X_test, y_train, _ = feature_engineering.split_and_encode(df)

        cols_before = set(X_train.columns)
        X_train_out, X_test_out, _ = correlation_analysis(
            X_train, y_train, X_test,
        )
        cols_after = set(X_train_out.columns)

        # Only the known drops should be removed
        expected_drops = {"TotalCharges", "_Household_Stability"}
        actual_drops = cols_before - cols_after
        assert actual_drops == expected_drops, (
            f"Unexpected feature drops: {actual_drops - expected_drops}"
        )
        # X_train and X_test must stay aligned
        assert list(X_train_out.columns) == list(X_test_out.columns)
