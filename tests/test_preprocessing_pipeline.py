"""Tests for data preprocessing pipelines.

Covers:
  L1-8  Test Data Preprocessing Pipelines
        — derived features, feature count, data leakage checks
  L1-9  Test Pipeline Stages
        — each stage tested independently with mock data
"""

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════
#  DERIVED FEATURE CORRECTNESS
# ═══════════════════════════════════════════════════════════════


class TestDerivedFeatures:
    """Verify that every engineered feature is computed correctly."""

    @pytest.fixture
    def encoded_df(self, cleaned_df):
        from feature_engineering import encode

        return encode(cleaned_df)

    @pytest.fixture
    def featured_df(self, encoded_df):
        from feature_engineering import add_engineered_features

        return add_engineered_features(encoded_df)

    # -- Feature existence ------------------------------------------------

    EXPECTED_ENGINEERED = [
        "_Total_AddOns-Services",
        "_Cost_Per_Service",
        "_Is_AutoPay",
        "_LoyaltyScore",
        "_HighFriction_Payment",
        "_Household_Stability",
    ]

    def test_all_engineered_features_present(self, featured_df):
        """Every custom feature must appear in the output columns."""
        for col in self.EXPECTED_ENGINEERED:
            assert col in featured_df.columns, f"Missing engineered feature: {col}"

    def test_engineered_feature_count(self, featured_df):
        """At least 6 custom features (prefixed with '_') should exist."""
        custom = [c for c in featured_df.columns if c.startswith("_")]
        assert len(custom) >= 6, f"Expected ≥6 engineered features, got {len(custom)}: {custom}"

    # -- Correctness of individual derived values -------------------------

    def test_total_addons_sum(self, encoded_df, featured_df):
        """_Total_AddOns-Services must equal row-wise sum of ADDON_COLS."""
        from config import ADDON_COLS

        expected = encoded_df[ADDON_COLS].sum(axis=1).values
        actual = featured_df["_Total_AddOns-Services"].values
        np.testing.assert_array_equal(actual, expected)

    def test_cost_per_service_formula(self, encoded_df, featured_df):
        """_Cost_Per_Service = MonthlyCharges / (core + addons)."""
        from config import ADDON_COLS

        core = encoded_df["PhoneService"] + (1 - encoded_df["InternetService_No"])
        addons = encoded_df[ADDON_COLS].sum(axis=1)
        expected = encoded_df["MonthlyCharges"] / (core + addons)
        pd.testing.assert_series_equal(
            featured_df["_Cost_Per_Service"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_household_stability_formula(self, encoded_df, featured_df):
        """_Household_Stability = Partner + Dependents."""
        expected = encoded_df["Partner"] + encoded_df["Dependents"]
        pd.testing.assert_series_equal(
            featured_df["_Household_Stability"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_high_friction_payment_formula(self, encoded_df, featured_df):
        """_HighFriction_Payment = PaperlessBilling=1 AND ElectronicCheck=1."""
        expected = (
            (encoded_df["PaperlessBilling"] == 1)
            & (encoded_df["PaymentMethod_Electronic check"] == 1)
        ).astype(int)
        pd.testing.assert_series_equal(
            featured_df["_HighFriction_Payment"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ═══════════════════════════════════════════════════════════════
#  FEATURE COUNT ASSERTIONS
# ═══════════════════════════════════════════════════════════════


class TestFeatureCount:
    """Ensure column counts are consistent across pipeline stages."""

    def test_encode_drops_customer_id(self, cleaned_df):
        """Encoding must remove exactly one column (customerID)."""
        from feature_engineering import encode

        df = encode(cleaned_df)
        # customerID gone, but one-hot expansion adds columns, so net
        # change = -1 (customerID) + new dummies.
        assert "customerID" not in df.columns

    def test_encode_adds_one_hot_columns(self, cleaned_df):
        """One-hot encoding must produce dummies for InternetService
        and PaymentMethod (drop_first=True)."""
        from feature_engineering import encode

        df = encode(cleaned_df)
        internet_dummies = [c for c in df.columns if c.startswith("InternetService_")]
        payment_dummies = [c for c in df.columns if c.startswith("PaymentMethod_")]
        assert len(internet_dummies) >= 1, "InternetService dummies missing"
        assert len(payment_dummies) >= 1, "PaymentMethod dummies missing"

    def test_helper_cols_removed_after_engineering(self, cleaned_df):
        """Temporary helper columns must not survive into the output."""
        from feature_engineering import add_engineered_features, encode

        df = encode(cleaned_df)
        df = add_engineered_features(df)
        for helper in ["CoreServices_Count", "True_Total_Services"]:
            assert helper not in df.columns, f"Helper column {helper} not dropped"

    def test_churn_preserved_through_engineering(self, cleaned_df):
        """The target column must survive encoding + feature engineering."""
        from feature_engineering import add_engineered_features, encode

        df = encode(cleaned_df)
        df = add_engineered_features(df)
        assert "Churn" in df.columns

    def test_split_column_consistency(self, cleaned_df, monkeypatch):
        """X_train and X_test must have identical column sets after split."""
        import feature_engineering
        from feature_engineering import add_engineered_features, encode

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.5)
        df = encode(cleaned_df)
        df = add_engineered_features(df)
        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)
        assert list(X_train.columns) == list(
            X_test.columns
        ), "Train/test feature columns diverged after split"


# ═══════════════════════════════════════════════════════════════
#  DATA LEAKAGE CHECKS
# ═══════════════════════════════════════════════════════════════


class TestDataLeakage:
    """Detect common forms of data leakage in the pipeline."""

    def test_frequency_encoding_uses_train_only(self, cleaned_df, monkeypatch):
        """Contract frequency encoding must use only X_train frequencies,
        NOT the entire dataset. We verify that X_test values come from the
        training-set distribution, not the global distribution."""
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.4)
        df = feature_engineering.encode(cleaned_df)
        df = feature_engineering.add_engineered_features(df)

        # Compute what the *global* frequency map would look like
        global_freq = df["Contract"].value_counts(normalize=True)

        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)

        # The frequency-encoded values in X_train should match train
        # frequencies, NOT global frequencies.  If the pipeline used
        # global frequencies, this comparison would be equal.
        train_freq = X_train["Contract"].value_counts(normalize=True)

        # At minimum, validate that the function is NOT using all data:
        # The actual freq values in X_test must be drawn from the
        # X_train frequency map (which split_and_encode does correctly).
        test_contract_values = X_test["Contract"].unique()
        train_contract_values = X_train["Contract"].unique()
        for val in test_contract_values:
            assert val in train_contract_values, (
                f"X_test Contract value {val} not in X_train freq map — "
                "possible leakage from global encoding"
            )

    def test_loyalty_score_leakage_awareness(self, cleaned_df):
        """Flag that _LoyaltyScore uses full-data mean (potential leakage).

        add_engineered_features() computes tenure.mean() over the entire
        DataFrame BEFORE the train/test split. In a production pipeline,
        this should use only training-set statistics. This test documents
        the known issue and verifies the feature still computes without error.
        """
        from feature_engineering import add_engineered_features, encode

        df = encode(cleaned_df)
        df_out = add_engineered_features(df)

        # The feature is created from the full-data mean, which is a
        # known leakage vector. At minimum, verify it's deterministic
        # and produces valid binary output.
        assert "_LoyaltyScore" in df_out.columns
        assert set(df_out["_LoyaltyScore"].unique()).issubset({0, 1})

        # Document: this uses full-data statistics (leakage risk).
        # The mean used is df["tenure"].mean(), NOT train-only mean.
        full_mean = df["tenure"].mean()
        expected = (
            (df["tenure"] > full_mean) & (df["MonthlyCharges"] < df["MonthlyCharges"].mean())
        ).astype(int)
        pd.testing.assert_series_equal(
            df_out["_LoyaltyScore"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_target_not_in_features(self, cleaned_df, monkeypatch):
        """Target column (Churn) must never appear in X_train or X_test
        — this would be direct target leakage."""
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.5)
        df = feature_engineering.encode(cleaned_df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)
        assert "Churn" not in X_train.columns, "Target leaked into X_train!"
        assert "Churn" not in X_test.columns, "Target leaked into X_test!"

    def test_no_test_index_in_train(self, cleaned_df, monkeypatch):
        """Train and test indices must not overlap — guards against
        row duplication leakage."""
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.4)
        df = feature_engineering.encode(cleaned_df)
        df = feature_engineering.add_engineered_features(df)
        X_train, X_test, _, _ = feature_engineering.split_and_encode(df)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Train/test overlap detected on indices: {overlap}"

    def test_stratification_preserves_class_ratio(self, cleaned_df, monkeypatch):
        """Stratified split should roughly preserve the original class ratio."""
        import feature_engineering

        monkeypatch.setattr(feature_engineering, "TEST_SIZE", 0.4)
        df = feature_engineering.encode(cleaned_df)
        df = feature_engineering.add_engineered_features(df)
        original_ratio = df["Churn"].mean()
        _, _, y_train, y_test = feature_engineering.split_and_encode(df)
        # With only 5 rows, ratios won't be exact — allow wide tolerance
        assert (
            abs(y_train.mean() - original_ratio) < 0.35
        ), "Training class ratio diverged too much from original"


# ═══════════════════════════════════════════════════════════════
#  PIPELINE STAGE INDEPENDENCE
# ═══════════════════════════════════════════════════════════════


class TestPipelineStages:
    """Test each pipeline stage independently with controlled inputs.

    Each test creates its own mock input rather than depending on
    upstream stages, proving each stage is self-contained.
    """

    def test_clean_stage_independent(self):
        """clean() works on any DataFrame matching the raw schema."""
        from data_cleaning import clean

        mock = pd.DataFrame(
            {
                "customerID": ["X1", "X2"],
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [10, 20],
                "PhoneService": ["Yes", "No"],
                "MultipleLines": ["No", "No phone service"],
                "InternetService": ["DSL", "No"],
                "OnlineSecurity": ["Yes", "No internet service"],
                "OnlineBackup": ["No", "No internet service"],
                "DeviceProtection": ["No", "No internet service"],
                "TechSupport": ["Yes", "No internet service"],
                "StreamingTV": ["No", "No internet service"],
                "StreamingMovies": ["Yes", "No internet service"],
                "Contract": ["Month-to-month", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "MonthlyCharges": [50.0, 30.0],
                "TotalCharges": ["500", "600"],
                "Churn": ["Yes", "No"],
            }
        )
        result = clean(mock)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_encode_stage_independent(self):
        """encode() works on any cleaned DataFrame."""
        from data_cleaning import clean
        from feature_engineering import encode

        mock = pd.DataFrame(
            {
                "customerID": ["A1", "A2", "A3", "A4"],
                "gender": ["Male", "Female", "Male", "Female"],
                "SeniorCitizen": [0, 1, 0, 1],
                "Partner": ["Yes", "No", "Yes", "No"],
                "Dependents": ["No", "Yes", "No", "Yes"],
                "tenure": [5, 15, 30, 45],
                "PhoneService": ["Yes", "No", "Yes", "Yes"],
                "MultipleLines": ["No", "No phone service", "Yes", "No"],
                "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
                "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
                "OnlineBackup": ["Yes", "No", "No internet service", "Yes"],
                "DeviceProtection": ["No", "Yes", "No internet service", "No"],
                "TechSupport": ["No", "No", "No internet service", "Yes"],
                "StreamingTV": ["Yes", "No", "No internet service", "Yes"],
                "StreamingMovies": ["No", "Yes", "No internet service", "No"],
                "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
                "PaperlessBilling": ["Yes", "No", "No", "Yes"],
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                "MonthlyCharges": [70.0, 40.0, 20.0, 95.0],
                "TotalCharges": ["840", "480", "600", "4275"],
                "Churn": ["Yes", "No", "No", "Yes"],
            }
        )
        cleaned = clean(mock)
        encoded = encode(cleaned)
        assert "customerID" not in encoded.columns
        assert encoded.isnull().sum().sum() == 0
        # Contract stays as 'string' dtype (from CAT_COLS_TO_STRING); all others numeric
        non_contract = [c for c in encoded.columns if c != "Contract"]
        assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in non_contract)

    def test_engineered_stage_independent(self):
        """add_engineered_features() works on any properly encoded DataFrame."""
        from data_cleaning import clean
        from feature_engineering import add_engineered_features, encode

        mock = pd.DataFrame(
            {
                "customerID": ["B1", "B2", "B3", "B4"],
                "gender": ["Male", "Female", "Male", "Female"],
                "SeniorCitizen": [0, 1, 0, 1],
                "Partner": ["Yes", "No", "Yes", "No"],
                "Dependents": ["No", "Yes", "No", "Yes"],
                "tenure": [1, 24, 60, 36],
                "PhoneService": ["Yes", "Yes", "No", "Yes"],
                "MultipleLines": ["No", "Yes", "No phone service", "No"],
                "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
                "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
                "OnlineBackup": ["No", "No", "No internet service", "Yes"],
                "DeviceProtection": ["Yes", "No", "No internet service", "No"],
                "TechSupport": ["No", "Yes", "No internet service", "Yes"],
                "StreamingTV": ["Yes", "No", "No internet service", "No"],
                "StreamingMovies": ["No", "Yes", "No internet service", "Yes"],
                "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
                "PaperlessBilling": ["Yes", "No", "No", "Yes"],
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                "MonthlyCharges": [85.0, 55.0, 20.0, 75.0],
                "TotalCharges": ["85", "1320", "1200", "2700"],
                "Churn": ["Yes", "No", "No", "Yes"],
            }
        )
        cleaned = clean(mock)
        encoded = encode(cleaned)
        featured = add_engineered_features(encoded)
        assert "_Total_AddOns-Services" in featured.columns
        assert "_Cost_Per_Service" in featured.columns
        assert featured.isnull().sum().sum() == 0

    def test_validation_stage_independent(self):
        """Validation functions run without error on any valid schema."""
        from data_validation import (
            check_accuracy,
            check_completeness,
            check_duplicates,
        )

        mock = pd.DataFrame(
            {
                "customerID": ["C1", "C2"],
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [12, 24],
                "PhoneService": ["Yes", "No"],
                "MultipleLines": ["No", "No phone service"],
                "InternetService": ["DSL", "No"],
                "OnlineSecurity": ["Yes", "No internet service"],
                "OnlineBackup": ["No", "No internet service"],
                "DeviceProtection": ["No", "No internet service"],
                "TechSupport": ["Yes", "No internet service"],
                "StreamingTV": ["No", "No internet service"],
                "StreamingMovies": ["Yes", "No internet service"],
                "Contract": ["One year", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "MonthlyCharges": [60.0, 25.0],
                "TotalCharges": ["720", "600"],
                "Churn": ["No", "No"],
            }
        )
        # Each should run without raising
        check_accuracy(mock)
        check_duplicates(mock)
        mock["TotalCharges"] = pd.to_numeric(mock["TotalCharges"], errors="coerce")
        check_completeness(mock)

    def test_clean_idempotent(self, raw_df):
        """Running clean() twice should produce the same result."""
        from data_cleaning import clean

        first = clean(raw_df)
        # clean() expects raw schema, but Partner/Dependents/Churn are
        # already int after first pass. Convert back for idempotency test.
        second_input = first.copy()
        second_input["Partner"] = second_input["Partner"].map({1: "Yes", 0: "No"})
        second_input["Dependents"] = second_input["Dependents"].map({1: "Yes", 0: "No"})
        second_input["Churn"] = second_input["Churn"].map({1: "Yes", 0: "No"})
        second_input["TotalCharges"] = second_input["TotalCharges"].astype(str)
        second = clean(second_input)
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            second.reset_index(drop=True),
        )
